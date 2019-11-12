from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, chi2
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.base import clone
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd


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
        cv_base_learners, cv_meta_learner = stacking(base_learners, clone(
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
