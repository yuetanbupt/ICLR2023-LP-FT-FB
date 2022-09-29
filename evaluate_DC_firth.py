import pickle
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from utils import LinearClassifier, torch_logistic_reg_lbfgs_batch
use_gpu = torch.cuda.is_available()

RATE = 1
def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    # calibrated_mean = 0.1 * mean[0] + 0.1 * mean[1] + 0.8 * mean[2]
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

if __name__ == '__main__':
    # ---- data loading
    dataset = 'miniImagenet'
    n_shot = 5
    n_ways = 10
    n_queries = 15
    n_runs = 1000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples


    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
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

    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))
    for i in tqdm(range(n_runs)):

        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- Tukey's transform
        beta = 0.5
        support_data = np.power(support_data[:, ] ,beta)
        query_data = np.power(query_data[:, ] ,beta)
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(750/n_shot)
        for i in range(n_lsamples):
            mean, cov = distribution_calibration(support_data[i], base_means, base_cov, k=2)
            features = np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled)
            # selected_features = select_features(support_data[i], features)
            sampled_data.append(features)
            sampled_label.extend([support_label[i]] * int(num_sampled*RATE))
        sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * int(num_sampled * RATE), -1)
        X_aug = np.concatenate([support_data, sampled_data])
        # X_aug = normalization(X_aug)
        Y_aug = np.concatenate([support_label, sampled_label])
        # ---- train classifier
        ###############################################################
        # classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
        # query_data = normalization(query_data)
        # predicts = classifier.predict(query_data)
        ###############################################################
        # linear_classifier = LinearClassifier(n_way = n_ways, n_support = n_shot)
        # scores = linear_classifier(X_aug, Y_aug, query_data)
        # scores = scores.detach().cpu().numpy()
        # predicts = np.argmax(scores, axis = -1)
        ################################################################
        X_aug = torch.FloatTensor(X_aug).unsqueeze(0).cuda()
        Y_aug = torch.LongTensor(Y_aug).unsqueeze(0).cuda()
        firth_linearclassifier = torch_logistic_reg_lbfgs_batch(X_aug, Y_aug, 0.0, 100, verbose=False)
        with torch.no_grad():
            query_data = torch.FloatTensor(query_data).cuda()
            query_label = torch.LongTensor(query_label).cuda()
            predicts = firth_linearclassifier(query_data).argmax(dim=-1)
            acc = (predicts == query_label).double().mean(dim=(-1)).detach().cpu().numpy().ravel()
        # acc = np.mean(predicts == query_label)
        acc_list.append(acc)
        print('%s %d way %d shot  ACC : %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))
    print('%s %d way %d shot  ACC : %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))

