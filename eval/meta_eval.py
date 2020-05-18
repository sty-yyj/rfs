from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from models import make_model, ContextBlock


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR'):
    net = net.eval()
    acc = []
    d_model = 640

    # fusion_module = make_model(1, d_model, 4 * d_model, 1, 0.1)
    fusion_module = ContextBlock(26, 2, fusion_types=('channel_mul', ))
    fusion_module.eval()
    ckpt = torch.load('fusion/gc*_ckpt_epoch_10.pth')
    fusion_module.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        fusion_module = fusion_module.cuda()
        cudnn.benchmark = True

    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            query_xs = query_xs.view(-1, height, width, channel)

            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            inputs = []
            for k in range(query_features.size(0)):
                inputs.append(torch.cat((query_features[k, None], support_features)))
            inputs = torch.stack(inputs)

            outs = fusion_module(inputs)
            # res = fusion_module(inputs)
            # outs = res[:, 0, :]
            # s_ys = res[:, 1:, :].mean(dim=0)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()
            outs = outs.detach().cpu().numpy()
            # s_ys = s_ys.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            if classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_ys)
                # clf.fit(s_ys, support_ys)
                # query_ys_pred = clf.predict(query_features)
                query_ys_pred = clf.predict(outs)
            elif classifier == 'NN':
                query_ys_pred = NN(support_features, support_ys, query_features)
                # query_ys_pred = NN(support_features, support_ys, outs)
            elif classifier == 'Cosine':
                # query_ys_pred = Cosine(support_features, support_ys, query_features)
                query_ys_pred = Cosine(support_features, support_ys, outs)
            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))

            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred
