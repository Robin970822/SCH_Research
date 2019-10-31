from data import loadData
from model import get_model

import config
import os

model_path = config.model_path

x_train = loadData(filename='x_train.npy')
y_train = loadData(filename='y_train.npy')

x_test = loadData(filename='x_test.npy')
y_test = loadData(filename='y_test.npy')

m, n = x_train.shape
print('N_features:', n)
# Training
model = get_model(input_shape=n, output_shape=2, model_type='MLP')
model.fit(x_train, y_train, epochs=1500, batch_size=64, validation_data=(
    x_test, y_test), validation_freq=100, verbose=1)

# Evaluate
print('Evaluating...')
model.evaluate(x_test, y_test)

# Save Model
print('Saving Model...')
model_path = os.path.join(model_path, 'model.h5')
model.save(model_path)
print('Model has Saved in {} \n Dataset {}'.format(model_path, len(y_train)))
