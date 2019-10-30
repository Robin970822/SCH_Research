import os
import h5py
import config

import numpy as np

root = config.scz_Data
data_path = config.data_path


def getPerson(filename='aal2_266.mat'):
    myfile = h5py.File(os.path.join(root, filename))
    data = [myfile[element[0]][:]
            for element in myfile[filename.split('.')[0]]]
    person = np.array([np.corrcoef(data_sample) for data_sample in data])
    return person


if __name__ == '__main__':
    person = getPerson()
    person_path = os.path.join(data_path, 'person.npy')
    print(person_path)
    np.save(person_path, person)
