import os
import h5py
import config
import scipy.io

import numpy as np

root = config.scz_Data
data_path = config.data_path


# 计算person相关矩阵
def getPerson(filename='aal2_266.mat'):
    myfile = h5py.File(os.path.join(root, filename))
    data = [myfile[element[0]][:]
            for element in myfile[filename.split('.')[0]]]
    person = np.array([np.corrcoef(data_sample) for data_sample in data])
    return person


# 获得标签
def getLabel(filename='all_cov.mat'):
    all_cov = scipy.io.loadmat(os.path.join(root, filename))
    all_cov = all_cov['all_cov']
    labels = all_cov[:, 0]
    return labels


# 获得头动数据
def getFD(filename='meanFD_info.mat'):
    meanFD = h5py.File(os.path.join(root, filename))
    meanFD = meanFD['meanFD'][0]
    return meanFD


# 排除头动数据
def select(labels, meanFD):
    index = [(label == 0 and fd < 0.2) or (label == 1 and fd < 0.3)
             for label, fd in zip(labels, meanFD)]
    return index


# 展开person矩阵上三角
def triu(person):
    temp = [person[i][i + 1: len(person)] for i in range(len(person))]
    return np.concatenate(temp)


def saveData(data, root=data_path, filename='default.npy'):
    path = os.path.join(root, filename)
    np.save(path, data)


if __name__ == '__main__':
    person = getPerson()
    labels = getLabel()
    fds = getFD()

    index = select(labels, fds)
    person = person[index]
    labels = labels[index]
    trius = [triu(a_person) for a_person in person]

    saveData(person, filename='person.npy')
    saveData(labels, filename='labels.npy')
    saveData(trius, filename='trius.npy')
