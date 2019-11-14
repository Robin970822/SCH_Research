from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from mlens.ensemble import SuperLearner
from data import loadData
from utils import feature_select

import config
import numpy as np


seed = config.seed


def get_models():
    nb = GaussianNB()
    svc = SVC(kernel='rbf', C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=10)
    lr = LogisticRegression(C=100, random_state=seed)
    nn = MLPClassifier((64, 16), solver='lbfgs',
                       activation='relu', random_state=seed)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=seed)
    rf = RandomForestClassifier(
        n_estimators=10, max_features=5, random_state=seed)
    ab = AdaBoostClassifier(random_state=seed)

    models = {'svm': svc, 'knn': knn, 'naive bayes': nb,
              'mlp': nn, 'random forest': rf, 'gbm': gb, 'logistic': lr,
              'adaboost': ab, }
    return models


def get_meta():
    meta_learner = GradientBoostingClassifier(
        n_estimators=1000,
        loss="exponential",
        max_features=4,
        max_depth=3,
        subsample=0.5,
        learning_rate=0.005,
        random_state=seed
    )
    return meta_learner


def get_ensemble():
    sl = SuperLearner(folds=10, random_state=seed,
                      verbose=2, backend='multiprocessing')
    sl.add(list(get_models().values()), proba=True)
    sl.add_meta(get_meta(), proba=True)
    return sl


def cross_val_ensemble(X, y, generator=KFold(10)):
    idx = list(range(len(X)))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    cv_scores = []
    for i, (train_idx, test_idx) in enumerate(generator.split(X)):
        fold_x_train, fold_y_train = X[train_idx, :], y[train_idx]
        fold_x_test, fold_y_test = X[test_idx, :], y[test_idx]

        sl = get_ensemble()
        sl.fit(fold_x_train, fold_y_train)
        fold_p = sl.predict_proba(fold_x_test)
        fold_y_pred = np.argmax(fold_p, axis=1)

        acc = accuracy_score(fold_y_test, fold_y_pred)
        recall = recall_score(fold_y_test, fold_y_pred)
        cm = confusion_matrix(fold_y_test, fold_y_pred)
        roc = roc_auc_score(fold_y_test, fold_p[:, 1])

        cv_scores.append({'acc': acc, 'recall': recall,
                          'confusion_matrix': cm, 'roc_auc_score': roc})
    return sl, cv_scores


if __name__ == '__main__':
    trius = loadData(filename='trius.npy')
    labels = loadData(filename='labels.npy')
    X = feature_select(trius, labels)
    y = labels

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=seed)
    print('样本集大小：', X.shape, y.shape)
    print('训练集大小：', x_train.shape, y_train.shape)  # 训练集样本大小
    print('测试集大小：', x_test.shape, y_test.shape)  # 测试集样本大小

    sl = get_ensemble()
    sl.fit(x_train, y_train)
    p_sl = sl.predict_proba(x_test)
    print("Super Learner ROC-AUC score: %.3f" %
          roc_auc_score(y_test, p_sl[:, 1]))
    print("Super Learner ACC score: %.3f" %
          accuracy_score(y_test, np.float32(p_sl[:, 1] >= 0.5)))

    model, cv_scores = cross_val_ensemble(X, y)

    acc = [i['acc'] for i in cv_scores]
    recall = [i['recall'] for i in cv_scores]
    roc = [i['roc_auc_score'] for i in cv_scores]

    print('Cross validation acc {} +/- {}'.format(np.mean(acc), np.std(acc)))
