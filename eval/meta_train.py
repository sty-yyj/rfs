from __future__ import print_function

import os
import sys
import time
import logging
import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from models import make_model, ContextBlock
from util import accuracy, AverageMeter, adjust_learning_rate


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


def meta_train(net, trainloader, use_logit=True, is_norm=True, classifier='LR'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log_file = 'fusion/' + time_str + '.log'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    # acc = []
    d_model = 640

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # fusion_module = make_model(1, d_model, 4 * d_model, 1, 0.1)
    fusion_module = ContextBlock(26, 2, fusion_types=('channel_mul', ))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(fusion_module.parameters(),
                          lr=0.05,
                          momentum=0.9,
                          weight_decay=5e-9)

    if torch.cuda.is_available():
        fusion_module.cuda()
        net.cuda()
        cudnn.benchmark = True

    opt = dict()
    opt["lr_decay_epochs"] = [60, 80]
    opt["learning_rate"] = 0.05
    opt["lr_decay_rate"] = 0.1

    # 100个epoch
    for i in range(100):
        # adjust_learning_rate(i+1, opt, optimizer)
        for idx, data in tqdm(enumerate(trainloader)):
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            query_ys = query_ys.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            query_xs = query_xs.view(-1, height, width, channel)

            feat_support, _ = net(support_xs, is_feat=True)
            support_features = feat_support[-1]
            feat_query, _ = net(query_xs, is_feat=True)
            query_features = feat_query[-1]

            inputs = []
            for k in range(query_features.size(0)):
                inputs.append(torch.cat((query_features[k, None], support_features)))
            inputs = torch.stack(inputs)

            outs = fusion_module(inputs)
            pred_querys = net.classifier(outs)

            query_ys = query_ys.view(-1)
            loss = loss_fn(pred_querys, query_ys)

            acc1, acc5 = accuracy(pred_querys, query_ys, topk=(1, 5))
            losses.update(loss.item(), query_xs.size(0))
            top1.update(acc1[0], query_xs.size(0))
            top5.update(acc5[0], query_xs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 50 == 0:
                logger.info('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(loss=losses, top1=top1, top5=top5))
                # sys.stdout.flush()

        print("Epoch:", i+1)
        if (i + 1) % 10 == 0:
            state = {
                'epoch': i+1,
                'model': fusion_module.state_dict()
            }
            save_file = os.path.join('fusion/cifar', 'gcckpt_epoch_{epoch}.pth'.format(epoch=i+1))
            torch.save(state, save_file)
        # if is_norm:
        #     support_features = normalize(support_features)
        #     query_features = normalize(query_features)

        # support_features_np = support_features.detach().cpu().numpy()
        # query_features_np = query_features.detach().cpu().numpy()

        # support_ys_np = support_ys.view(-1).numpy()
        # query_ys_np = query_ys.view(-1).numpy()

        # if classifier == 'LR':
        #     clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
        #                             multi_class='multinomial')
        #     clf.fit(support_features_np, support_ys_np)
        #     query_ys_pred = clf.predict(query_features_np)
        # elif classifier == 'NN':
        #     query_ys_pred = NN(support_features, support_ys, query_features)
        # elif classifier == 'Cosine':
        #     query_ys_pred = Cosine(support_features, support_ys, query_features)
        # else:
        #     raise NotImplementedError('classifier not supported: {}'.format(classifier))

        # acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    # return mean_confidence_interval(acc)


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
