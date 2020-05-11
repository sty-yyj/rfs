# from sklearn.linear_model import LogisticRegression
# import numpy as np


# support_features = np.load('s.npy')
# support_ys = np.load('s_y.npy')
# query_features = np.load('q.npy')
# query_ys = np.load('q_y.npy')

# clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
#                             multi_class='multinomial')
# clf.fit(support_features, support_ys)
# query_ys_pred = clf.predict(query_features)

# # indexes = np.where(query_ys != query_ys_pred)
# # print(indexes)
# query_feature = query_features[9, None]
# support_feature = support_features[5, None]

# query_feature = 0.5 * query_feature + 0.5 * support_feature
# y = clf.predict(query_feature)
# print(y)
import logging
import time

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
for i in range(10):
    logger.info("testing")