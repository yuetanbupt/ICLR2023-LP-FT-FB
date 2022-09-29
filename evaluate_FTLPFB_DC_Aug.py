import pickle
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from utils import LinearClassifier, FineTuning, generate_dataloader, WrappedModel
import wrn_model
import configs
from save_plk import extract_feature
from data.datamgr import SimpleDataManager

use_gpu = torch.cuda.is_available()

RATE = 1
def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov

def select_features(query, features, rate = RATE):
    dist = []
    k = int(len(features) * rate)
    for i in range(len(features)):
        dist.append(np.linalg.norm(query-features[i]))
    index = np.argpartition(dist, k)[:k]
    selected_features = features[index]
    return selected_features

def normalization(features):
    features_norm = np.linalg.norm(features, axis = 1, ord = 2, keepdims = True)
    features = features / features_norm
    return features

def load_model(F_model, F_path, dataset = 'miniImagenet'):
    F_init_parameter = torch.load(F_path)
    state = F_init_parameter['state']
    state_keys = list(state.keys())
    if dataset == 'tiered_imagenet' or dataset == 'CUB':
        for key in state_keys:
            if 'classifier' in key:
                new_key = key.replace('classifier', 'linear')
                state[new_key] = state.pop(key)
        callwrap = False
        if 'module' in state_keys[0]:
            callwrap = True
        if callwrap:
            F_model = WrappedModel(F_model)

    model_dict_load = F_model.state_dict()
    model_dict_load.update(state)

    F_model.load_state_dict(model_dict_load)
    F_model.cuda()
    return F_model

def truelabel2artificial(truelabel, unique_label = None):
    if unique_label is None:
        unique_label = torch.unique(truelabel)
    for index, label in enumerate(unique_label):
        indices = torch.nonzero(torch.eq(truelabel, label))
        truelabel[indices] = index
    return unique_label, truelabel

def extract_base_features(dataset, F_model_finetuning, checkpoint_dir):
    loadfile_base = configs.data_dir[dataset] + 'base.json'
    datamgr = SimpleDataManager(84, batch_size = 256)
    base_loader = datamgr.get_data_loader(loadfile_base, aug=False)
    output_dict_base = extract_feature(base_loader, F_model, checkpoint_dir, tag='tmp', set='base')

if __name__ == '__main__':
    # ---- data loading
    dataset = 'miniImagenet'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 1000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    num_classes = {'miniImagenet': 200, 'tiered_imagenet': 351, 'CUB': 200}
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, dataset, 'WideResNet28_10', 'S2M2_R')
    '''
    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    print(ndatas.shape)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs, n_samples)
    '''
    dataloader = generate_dataloader(n_ways, n_shot, n_queries, n_runs)
    # ---- Base class statistics
    '''
    base_means = []
    base_cov = []
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)
    '''

    # ---- classification for each task
    F_model = wrn_model.wrn28_10(num_classes=num_classes[dataset])
    F_path = './checkpoints/%s/WideResNet28_10_S2M2_R/470.tar'%dataset
    H_path = './checkpoints/%s/WideResNet28_10_S2M2_R/linear_layer.pth'%dataset
    F_model = load_model(F_model, F_path, dataset)
    for firth_c in [1]:
        acc_list = []
        acc_wo_finetune_list = []
        acc_aug_list = []
        print('Start classification for %d tasks...'%(n_runs))
        for i, (ndatas, labels) in tqdm(enumerate(dataloader)):
            nc, ns, c, h, w = ndatas.shape
            support_data = ndatas[:, :n_shot].reshape(nc * n_shot, c, h, w).cuda()
            support_label = labels[:, :n_shot].reshape(nc * n_shot)

            unique_label, support_label = truelabel2artificial(support_label)
            support_label = support_label.numpy()
            query_data = ndatas[:, n_shot:].reshape(nc * n_queries, c, h, w).cuda()
            query_label = labels[:, n_shot:].reshape(nc * n_queries)
            _, query_label = truelabel2artificial(query_label, unique_label)
            query_label = query_label.numpy()

            support_features = F_model(support_data)[0].detach().cpu().numpy()
            query_features = F_model(query_data)[0].detach().cpu().numpy()
            # ---- Tukey's transform
            beta = 0.5
            # support_features = np.power(support_features[:, ] ,beta)
            # query_features = np.power(query_features[:, ] ,beta)
            # ---- distribution calibration and feature sampling
            # X_aug = np.concatenate([support_features, sampled_data])
            X_aug = support_features
            Y_aug = support_label
            # ---- train classifier
            #classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
            # predicts = classifier.predict(query_data)
            ##########################################################################################################
            linear_classifier = LinearClassifier(n_way = n_ways, n_support = n_shot, batch = 4)
            scores = linear_classifier(X_aug, Y_aug, query_features, dataset = dataset, firth_c = 1)#, dataset = dataset)
            scores = scores.detach().cpu().numpy()
            predicts_wo_finetune = np.argmax(scores, axis = -1)
            acc_wo_finetune = np.mean(predicts_wo_finetune == query_label)
        
            H_model = linear_classifier.linear_clf
            finetuning = FineTuning(F_model, H_model, F_path, H_path, dataset = dataset)
            finetuning.train_loop(support_data, torch.tensor(support_label), firth_c = 0.001)
            acc = finetuning.test_loop(query_data, query_label)
            
            F_model_finetune = finetuning.F_model
            extract_base_features(dataset, F_model_finetune, checkpoint_dir)
            # ---- Base class statistics
            base_means = []
            base_cov = []
            base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/tmp/base_features.plk"%dataset
            with open(base_features_path, 'rb') as f:
                data = pickle.load(f)
                for key in data.keys():
                    feature = np.array(data[key])
                    mean = np.mean(feature, axis=0)
                    cov = np.cov(feature.T)
                    base_means.append(mean)
                    base_cov.append(cov)
            ################################################################################
            support_features, _ = F_model_finetune(support_data)
            query_features, _ = F_model_finetune(query_data)
            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()
            support_features = np.power(support_features[:, ] ,beta)
            query_features = np.power(query_features[:, ] ,beta)
            sampled_data = []
            sampled_label = []
            num_sampled = int(750/n_shot)
            for i in range(n_lsamples):
                mean, cov = distribution_calibration(support_features[i], base_means, base_cov, k=2)
                features = np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled)
                sampled_data.append(features)
                sampled_label.extend([support_label[i]] * int(num_sampled*RATE))
            sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * int(num_sampled * RATE), -1)
            X_aug = np.concatenate([support_features, sampled_data])
            Y_aug = np.concatenate([support_label, sampled_label])
            
            classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
            predicts = classifier.predict(query_features)
            # acc = finetuning.test_loop(query_data, query_label)
            acc_aug = np.mean(predicts == query_label)
    
            acc_list.append(acc)
            acc_wo_finetune_list.append(acc_wo_finetune)
            acc_aug_list.append(acc_aug)
            print('%s %d way %d shot  ACC + LP + FBR: %f'%(dataset,n_ways,n_shot,float(np.mean(acc_wo_finetune_list))))
            print('%s %d way %d shot  ACC + LPFTFB: %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))
            print('%s %d way %d shot  ACC + LPFTFB + DC: %f'%(dataset,n_ways,n_shot,float(np.mean(acc_aug_list))))

        print('%s %d way %d shot  ACC + LP + FBR: %f'%(dataset,n_ways,n_shot,float(np.mean(acc_wo_finetune_list))))
        print('%s %d way %d shot  ACC + LPFTFB: %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))
        print('%s %d way %d shot  ACC + LPFTFB + DC: %f'%(dataset,n_ways,n_shot,float(np.mean(acc_aug_list))))
