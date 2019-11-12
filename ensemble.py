from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, chi2
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve  # 模型度量
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
from mlens.ensemble import SuperLearner
from matplotlib import pyplot as plt
#

import config
import numpy as np
import pandas as pd


seed = config.seed


def feature_select(X, y):
    X = MinMaxScaler().fit_transform(X)

    selector = VarianceThreshold(0.03)
    selector.fit(X)
    X = selector.transform(X)

    selector = SelectKBest(chi2, k=2048)
    X = selector.fit_transform(X, y)

    etc = GradientBoostingClassifier().fit(X, y)
    model = SelectFromModel(etc, prefit=True)
    X = model.transform(X)

    X = PCA(n_components=128).fit_transform(X)

    return X


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


def train_base_learners(base_learners, x_train, y_train):
    print('Fitting models')
    for i, (name, m) in enumerate(base_learners.items()):
        print('%s...' % name, end='', flush=False)
        m.fit(x_train, y_train)
    print('done.')


def predict_base_learners(base_learners, x):
    P = np.zeros((x.shape[0], len(base_learners)))
    print('Generating base learner predictions.')
    for i, (name, m) in enumerate(base_learners.items()):
        print('%s...' % name, end='', flush=False)
        p = m.predict_proba(x)
        P[:, i] = p[:, 1]
    print('done.')
    return P


def train_predict(x_train, y_train, x_test, y_test, model_list):
    P = np.zeros((y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print('Fitting models...')
    cols = list()
    for i, (name, m) in enumerate(model_list.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(x_train, y_train)
        P.iloc[:, i] = m.predict_proba(x_test)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def ensemble_predict(base_learners, meta_learner, x):
    P_pred = predict_base_learners(base_learners, x)
    return P_pred, meta_learner.predict_proba(P_pred)[:, 1]


def stacking(base_learners, meta_learner, X, y, generator=KFold(10)):
    print('Fitting base learners...', end='')
    train_base_learners(base_learners, X, y)
    print('done.')

    print('Generating cross-validated predctions...')
    cv_preds, cv_y = [], []
    for i, (train_idx, test_idx) in enumerate(generator.split(X)):
        fold_x_train, fold_y_train = X[train_idx, :], y[train_idx]
        fold_x_test, fold_y_test = X[test_idx, :], y[test_idx]
        fold_base_learners = {name: clone(model)
                              for name, model in base_learners.items()}
        train_base_learners(fold_base_learners, fold_x_train, fold_y_train)
        fold_P_base = predict_base_learners(fold_base_learners, fold_x_test)
        cv_preds.append(fold_P_base)
        cv_y.append(fold_y_test)
        print('Fold %i done' % i)
    cv_preds = np.vstack(cv_preds)
    cv_y = np.hstack(cv_y)

    print('Fitting meta learner...', end='')
    meta_learner.fit(cv_preds, cv_y)
    print('done')
    return base_learners, meta_learner


def cross_val_stacking(base_learners, meta_learner, X, y, generator=KFold(10)):
    cv_scores = []
    for i, (train_idx, test_idx) in enumerate(generator.split(X)):
        fold_x_train, fold_y_train = X[train_idx, :], y[train_idx]
        fold_x_test, fold_y_test = X[test_idx, :], y[test_idx]
        cv_base_learners, cv_meta_learner = stacking(get_models(), clone(
            meta_learner), fold_x_train, fold_y_train, generator)
        _, p = ensemble_predict(cv_base_learners, cv_meta_learner, fold_x_test)
        acc = accuracy_score(fold_y_test, np.float32(p > 0.5))
        cv_scores.append(acc)
    return cv_scores


def plot_roc_curve(y_test, P_base_learners, P_ensemble, labels, ens_label):
    plt.plot([0, 1], [0, 1], 'k--')
    cm = [plt.cm.rainbow(i) for i in np.linspace(
        0, 1.0, P_base_learners.shape[1] + 1)]

    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    fpr, tpr, _ = roc_curve(y_test, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()


def score_models(P, y):
    print('ROC AUC SCORE')
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print('ACC SCORE')
    for m in P.columns:
        score = accuracy_score(y, np.float32(P.loc[:, m] >= 0.5))
        print("%-26s: %.3f" % (m, score))


if __name__ == '__main__':
    from data import loadData
    trius = loadData('trius.npy')
    labels = loadData('label.npy')
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

    model,  cv_scores = cross_val_ensemble(X, y)

    acc = [i['acc'] for i in cv_scores]
    recall = [i['recall'] for i in cv_scores]
    roc = [i['roc_auc_score'] for i in cv_scores]

    print('Cross validation acc {} +/- {}'.format(np.mean(acc), np.std(acc)))
