from sklearn.decomposition import NMF
from data import loadData, saveData
import numpy as np


# 最重要的参数是n_components、alpha、l1_ratio、solver
nmf = NMF(n_components=128,  # k value,默认会保留全部特征
          # W H 的初始化方法，'random' | 'nndsvd'(默认) |  'nndsvda' | 'nndsvdar' | 'custom',
          init=None,
          solver='cd',  # 'cd' | 'mu'
          # {'frobenius', 'kullback-leibler', 'itakura-saito'},
          beta_loss='frobenius',
          tol=1e-10,  # 停止迭代的极限条件
          max_iter=200,  # 最大迭代次数
          random_state=None,
          alpha=0.,  # 正则化参数
          l1_ratio=0.,  # 正则化参数
          verbose=0,  # 冗长模式
          shuffle=False  # 针对"cd solver"
          )

trius = loadData(filename='trius.npy')
X = np.abs(trius)
nmf.fit(X)
W = nmf.fit_transform(X)
H = nmf.components_
print('reconstruction_err_', nmf.reconstruction_err_)  # 损失函数值
print('n_iter_', nmf.n_iter_)  # 实际迭代次数
saveData(W, filename='nmf.npy')
saveData(H, filename='basis.npy')
