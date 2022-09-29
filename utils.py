import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
from torch.optim import LBFGS
import configs
from data.datamgr import SimpleDataManager, SetDataManager
import copy
import torch.nn.functional as F
  
def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

def Proto_classifier(support_data, support_label, query_data, metric_type = 'euclidean'):
    support_data, proto = get_proto(support_data, support_label)
    m, d = proto.shape
    n, d = query_data.shape

    proto = proto[np.newaxis, :, :]
    query_data = query_data[:, np.newaxis, :]
    proto_expand = np.tile(proto, (n, 1, 1))
    query_data_expand = np.tile(query_data, (1, m, 1))

    diff = proto_expand - query_data_expand
    diff = np.transpose(diff, (1, 0, 2))
    if metric_type == 'euclidean':
        Sigma = np.eye(d)

    else:
        Sigma_dct = get_Maha_Sigma(np.squeeze(proto))
        Sigma = np.stack(list(Sigma_dct.values()), axis = 0)

    dists = np.transpose(((diff @ Sigma) * diff).sum(2), (1, 0))
    scores = -dists
    return scores

def get_proto(support_data, support_label, n_way = 5):
    n, d = support_data.shape
    label_unique = np.unique(support_label)
    support_rebuilt = []
    for label in label_unique:
        indices = np.nonzero(support_label == label)
        support_rebuilt.append(support_data[indices])
    support_rebuilt = np.reshape(np.concatenate(support_rebuilt, axis = 0), \
                                (n_way, n // n_way, d))
    proto = np.mean(support_rebuilt, axis = 1)
    return support_rebuilt, proto

def get_Maha_Sigma(proto):
    task_covariance_estimate = estimate_cov(proto)
    class_precision_matrices = {}
    for index, cl_proto in enumerate(proto):
        lambda_k_tau = 5 / (5 + 1)
        class_precision_matrices[index] = \
                np.linalg.inv((lambda_k_tau * estimate_cov(cl_proto)) \
                + (lambda_k_tau * task_covariance_estimate)\
                + np.eye(cl_proto.shape[0]))
    return class_precision_matrices


def estimate_cov(examples, rowvar=False, inplace=False):
    if len(examples.shape) > 2:
        raise ValueError('m has more than 2 dimensions')
    if len(examples.shape) < 2:
        examples = np.reshape(examples, (1, -1))
    if not rowvar and examples.shape[0] != 1:
        examples = examples.T
    factor = 1.0 / (examples.shape[1] - 1)
    if inplace:
        examples -= np.mean(examples, axis=1, keepdims=True)
    else:
        examples = examples - np.mean(examples, axis=1, keepdims=True)
    examples_t = examples.T
    return np.squeeze(factor * (examples @ examples_t))

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2;
        else:
            self.scale_factor = 10;

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor* (cos_dist)

        return scores


class LinearClassifier(nn.Module):
    def __init__(self, n_way, n_support, lr = 0.01, batch = 256, epoch_num = 300, loss_type = 'dist', save_tar = True):
        super(LinearClassifier, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.lr = lr
        self.batch = batch
        self.epoch_num = epoch_num
        self.loss_type = loss_type
        if self.loss_type == 'softmax':
            self.linear_clf = nn.Linear(640, self.n_way).cuda()
        elif self.loss_type == 'dist':
            self.linear_clf = distLinear(640, self.n_way).cuda()
        self.save_tar = save_tar

    def forward(self, support_z, support_y, query_z, save_path = None, firth_c = None, dataset = 'miniImagenet'):
        support_size, d = support_z.shape
        support_z = torch.from_numpy(support_z).float().cuda()
        support_y = torch.from_numpy(support_y).cuda()
        query_z = torch.from_numpy(query_z).float().cuda()

        optimizer = torch.optim.SGD(self.linear_clf.parameters(), lr = self.lr, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss().cuda()

        batch_size = self.batch
        for epoch in range(self.epoch_num):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = support_z[selected_id]
                y_batch = support_y[selected_id]
                scores = self.linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                if firth_c:
                    logp_tilde = scores
                    logp_hat = logp_tilde - torch.logsumexp(logp_tilde, axis=1, keepdim=True)
                    firth_term = logp_hat.mean()
                    loss_firth = (-firth_c) * firth_term
                    loss = loss + loss_firth
                # loss = loss_function(y_batch, scores)
                loss.backward()
                optimizer.step()
        if self.save_tar:
            if save_path is None:
                save_path = './checkpoints/%s/WideResNet28_10_S2M2_R/linear_layer.pth'%dataset
            self.save_parameters(self.linear_clf, save_path)
        scores = self.linear_clf(query_z)
        return scores

    def one_hot(self, target, class_num):
        batch_size = target.shape[0]
        return torch.zeros(batch_size,class_num).cuda().scatter_(1,target,1)

    def save_parameters(self, linear_layer, save_path):
        torch.save(linear_layer.state_dict(), save_path)

def select_features(query, features, rate = 0.2):
    dist = []
    k = int(len(features) * rate)
    for i in range(len(features)):
        dist.append(np.linalg.norm(query-features[i]))
    index0 = np.argpartition(dist, -k)[-k:]
    index1 = np.argpartition(dist, k)[:k]
    selected_features0 = features[index0]
    selected_features1 = features[index1]
    selected_features = np.concatenate([selected_features0, selected_features1],axis = 0)
    return selected_features

def torch_logistic_reg_lbfgs_batch(X_aug, Y_aug, firth_c=0.0, max_iter=1000, verbose=True, loss_type = 'softmax'):

    batch_dim, n_samps, n_dim = X_aug.shape
    assert Y_aug.shape == (batch_dim, n_samps)
    num_classes = Y_aug.unique().numel()

    device = X_aug.device
    tch_dtype = X_aug.dtype
    # default value from https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

    # from scipy.minimize.lbfgsb. In pytorch, it is the equivalent "max_iter"
    # (note that "max_iter" in torch.optim.LBFGS is defined per epoch and a step function call!)
    max_corr = 10
    tolerance_grad = 1e-05
    tolerance_change = 1e-09
    line_search_fn = 'strong_wolfe'
    l2_c = 1.0
    use_bias = True

    # According to https://github.com/scipy/scipy/blob/master/scipy/optimize/_lbfgsb_py.py#L339
    # L-BFGS optimization algorithm
    # wa (i.e., the equivalenet of history_size) is 2 * m * n (where m is max_corrections and n is the dimensions).
    history_size = max_corr * 2  # since wa is O(2*m*n) in size

    num_epochs = max_iter // max_corr  # number of optimization steps
    max_eval_per_epoch = None  # int(max_corr * max_evals / max_iter) matches the 15000 default limit in scipy!

    W = torch.nn.Parameter(torch.zeros((batch_dim, n_dim, num_classes), device=device, dtype=tch_dtype))
    opt_params = [W]
    linlayer = lambda x_: x_.matmul(W)
    if use_bias:
        bias = torch.nn.Parameter(torch.zeros((batch_dim, 1, num_classes), device=device, dtype=tch_dtype))
        opt_params.append(bias)
        linlayer = lambda x_: (x_.matmul(W) + bias)
    '''
    if loss_type == 'softmax':
        linlayer = nn.Linear(n_dim, num_classes, bias = False).cuda()
        W = dict(linlayer.named_parameters())['weight']
        opt_params = linlayer.parameters()
    else:
        linlayer = distLinear(n_dim, num_classes).cuda()
        W = dict(linlayer.L.named_parameters())['weight_v']
        opt_params = linlayer.L.parameters()
    '''
    optimizer = LBFGS(opt_params, lr=1, max_iter=max_corr, max_eval=max_eval_per_epoch,
                      tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                      history_size=history_size, line_search_fn=line_search_fn)

    Y_aug_i64 = Y_aug.to(device=device, dtype=torch.int64)
    for epoch in range(num_epochs):
        if verbose:
            running_loss = 0.0

        inputs_, labels_ = X_aug, Y_aug_i64

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            batch_dim_, n_samps_, n_dim_ = inputs_.shape
            outputs_ = linlayer(inputs_)
            # outputs_.shape -> batch_dim, n_samps, num_classes
            logp = outputs_ - torch.logsumexp(outputs_, dim=-1, keepdims=True)
            # logp.shape -> batch_dim, n_samps, num_classes
            label_logps = -logp.gather(dim=-1, index=labels_.reshape(batch_dim_, n_samps_, 1))
            # label_logps.shape -> batch_dim, n_samps, 1
            loss_cross = label_logps.mean(dim=(-1, -2)).sum(dim=0)
            loss_firth = -logp.mean(dim=(-1, -2)).sum(dim=0)
            loss_l2 = 0.5 * torch.square(W).sum() / n_samps_

            loss = loss_cross + firth_c * loss_firth + l2_c * loss_l2
            loss = loss / batch_dim_
            if loss.requires_grad:
                loss.backward()
            return loss

        # Update weights
        optimizer.step(closure)

        # Update the running loss
        if verbose:
            loss = closure()
            running_loss += loss.item()
            logger(f"Epoch: {epoch + 1:02}/{num_epochs} Loss: {running_loss:.5e}")
    return linlayer

class GradientDescentLearningRule(nn.Module):
    def __init__(self, device, learning_rate=1e-3):
        super(GradientDescentLearningRule, self).__init__()
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.learning_rate = torch.ones(1).cuda() * learning_rate

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, tau=0.9):
        return {
            key: names_weights_dict[key]
            - self.learning_rate * names_grads_wrt_params_dict[key]
            for key in names_weights_dict.keys()
        }

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
    def forward(self, x):
        return self.module(x)

class FineTuning(nn.Module):
    def __init__(self, F_model, H_model, F_path, H_path, lr = 0.001, epoch_num = 300, n_shot = 1, n_ways = 5, batch_size = 4, dataset = 'miniImagenet'):
        super(FineTuning, self).__init__()
        self.F_path = F_path
        self.H_path = H_path
        self.F_model = F_model
        self.H_model = H_model
        self.inner_loop_optimizer = GradientDescentLearningRule(device = 'cuda:0', learning_rate = lr)
        self.loss_function = nn.CrossEntropyLoss().cuda()
        # self.loss_function = LabelSmoothingCrossEntropy(reduction='sum')
        self.epoch_num = epoch_num
        self.batch_size = 4
        self.dataset = dataset

    def load_model(self):
        F_init_parameter = torch.load(self.F_path)
        state = F_init_parameter['state']
        state_keys = list(state.keys())
        if self.dataset == 'tiered_imagenet' or self.dataset == 'CUB':
            for key in state_keys:
                if 'classifier' in key:
                    new_key = key.replace('classifier', 'linear')
                    state[new_key] = state.pop(key)
            '''
            callwrap = False
            if 'module' in state_keys[0]:
                callwrap = True
            if callwrap:
                self.F_model = WrappedModel(self.F_model)
            '''

        model_dict_load = self.F_model.state_dict()
        model_dict_load.update(state)

        self.F_model.load_state_dict(model_dict_load)
        self.F_model.cuda()
        H_init_parameter = torch.load(self.H_path)
        self.H_model.load_state_dict(H_init_parameter)
        self.H_model.cuda()

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param.cuda()
            for name, param in params
            if param.requires_grad
            and "bn" not in name
        }

    def update_parameters(self, loss, names_weights_copy, target = 'features'):
        if target == 'features':
            if self.dataset == 'tiered_imagenet' or self.dataset == 'CUB':
                self.F_model.module.zero_grad(params = names_weights_copy)
            else:
                self.F_model.zero_grad(params = names_weights_copy)
            retain_graph = True
        else:
            self.H_model.zero_grad()
            retain_graph = False
        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=False, allow_unused=True, 
                                    retain_graph=retain_graph)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        # names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if key == 'linear.L.weight_g' or key == 'linear.L.weight_v':
                # names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                names_grads_copy[key] = torch.zeros_like(names_weights_copy[key]).cuda()

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy, names_grads_wrt_params_dict=names_grads_copy)

        '''
        num_devices = 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}
        '''

        return names_weights_copy

    def write_updated_params(self, names_weights_copy, target = 'features'):
        if target == 'features':
            model = self.F_model
        else:
            model = self.H_model

        model_dct = dict(model.named_parameters())
        for key, value in names_weights_copy.items():
            if (self.dataset == 'tiered_imagenet' or self.dataset == 'CUB') and target == 'features':
                new_key = 'module.' + key
                model_dct[new_key].data = value
            else:
                model_dct[key].data = value

    def train_loop(self, inputs, labels, firth_c = None):
        self.load_model()

        inputs = inputs.cuda()
        labels = labels.cuda()
        support_size = inputs.shape[0]
        for epoch in range(self.epoch_num):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , self.batch_size):
                selected_id = torch.from_numpy( rand_id[i: min(i+self.batch_size, support_size) ]).cuda()
                z_batch = inputs[selected_id]
                y_batch = labels[selected_id]

                # for name, params in self.F_model.named_parameters():
                #     print(name, params.shape)

                features, _ = self.F_model(z_batch)
                # features = torch.pow(features, 0.5)
                logits = self.H_model(features)
                loss = self.loss_function(logits, y_batch)
                if firth_c:
                    logp_tilde = logits
                    logp_hat = logp_tilde - torch.logsumexp(logp_tilde, axis=1, keepdim=True)
                    firth_term = logp_hat.mean()
                    loss_firth = (-firth_c) * firth_term
                    loss = loss + loss_firth
                if self.dataset == 'tiered_imagenet' or self.dataset == 'CUB':
                    names_weights_dict_F = self.get_inner_loop_parameter_dict(self.F_model.module.named_parameters())
                else:
                    names_weights_dict_F = self.get_inner_loop_parameter_dict(self.F_model.named_parameters())
                names_weights_dict_H = self.get_inner_loop_parameter_dict(self.H_model.named_parameters())
                names_weights_copy_F = self.update_parameters(loss, names_weights_dict_F)
                names_weights_copy_H = self.update_parameters(loss, names_weights_dict_H, 'head')
                self.write_updated_params(names_weights_copy_F, 'features')
                self.write_updated_params(names_weights_copy_H, 'head')

    def test_loop(self, query_data, query_labels):
        # query_data = torch.FloatTensor(query_data).cuda()
        features, _ = self.F_model(query_data)
        logits = self.H_model(features).detach().cpu().numpy()
        predicts = np.argmax(logits, axis = -1)
        acc = np.mean(predicts == query_labels)
        return acc

    def compute_diff(self, tmp, target = 'features'):
        if target == 'features':
            print('-------------------------------------------')
            params_dict = dict(self.F_model.named_parameters())
        else:
            print('===========================================')
            params_dict = dict(self.H_model.named_parameters())

        diff = 0
        for key in tmp:
            diff += torch.sum(tmp[key] - params_dict[key]).item()

        print(diff)

def generate_dataloader(n_ways, n_shot, n_queries, iter_num, dataset = 'miniImagenet', batch_size = 64):
    image_size = 84
    split = 'novel'
    iter_num = iter_num
    loadfile   = configs.data_dir[dataset] + split +'.json'
    few_shot_params = dict(n_way = n_ways, n_support = n_shot)
    datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = n_queries, **few_shot_params)
    data_loader      = datamgr.get_data_loader(loadfile, aug = False)
    return data_loader
