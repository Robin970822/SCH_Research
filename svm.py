from sklearn import svm
from sklearn.model_selection import cross_val_score
from data import loadData


x_train = loadData(filename='train_encode.npy')
y_train = loadData(filename='y_train.npy')

x_test = loadData(filename='test_encode.npy')
y_test = loadData(filename='y_test.npy')

m, n = x_train.shape
print('N_features:', n)
# Training
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, x_train, y_train, cv=5)
print('准确率：', scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
