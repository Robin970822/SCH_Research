import os


root = './'
seed = 2019

def get_path(root=root, path='model'):
    model_path = os.path.join(root, path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    return model_path


scz_Data = get_path(path='scz_Data')
data_path = get_path(path='data')
model_path = get_path(path='model')
