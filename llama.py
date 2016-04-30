from collections import Counter

import pandas as pd
import numpy
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import f1_score 

import xgboost as xgb


def load_df(train_path, test_path):
    train_df = pd.read_csv(train_path)
    print("Train DataFrame:")
    print(train_df.columns)
    print(train_df.shape)
    test_X = pd.read_csv(test_path)
    print("---\nTest DataFrame:")
    print(test_X.columns)
    print(test_X.shape)
    return train_df, test_X


def split_train_set(train_df, target_col):
    train_X = train_df.drop(target_col, 1)
    train_y = train_df[target_col]
    return train_X, train_y
    

def maxminscale(col_data, max_val, min_val):
    return (col_data - min_val) / (max_val - min_val)


def normalise(train, test, columns):
    for c in columns:
#         norm = MinMaxScaler()
#         norm.fit(numpy.concatenate([train[c], test[c]]))
        max_val, min_val = numpy.concatenate([train[c], test[c]]).max(), numpy.concatenate([train[c], test[c]]).min()
        train[c] = maxminscale(train[c], max_val, min_val)
        test[c] = maxminscale(test[c], max_val, min_val)


def insert_predictions(df, pred, colname):
    for col_i in range(pred.shape[1]):
        df.insert(len(df.columns), colname + str(col_i), pred[:,col_i])
        

def downsample(y, sizes = [30000, 3000]):
#     classes = Counter(y)
    res = []
    for class_i, sz in enumerate(sizes):
        indices = [x for x in y == class_i if x]
        res.append(resample(indices, replace = True, n_samples = sz))
    return tuple(res)
        

def tree_best_features(X, y, top = 30):
    rf = ExtraTreesClassifier(n_estimators=100)
    rf.fit(X, y)
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
    feat_imp.sort_values(inplace=True)
    ax = feat_imp.tail(top).plot(kind='barh', figsize=(10,7), title='Feature importance (random forest)')
    return feat_imp

    
def anova_best_features(X, y, top = 30):
    kbest = SelectKBest(f_classif)
    kbest.fit(X, y)
    feat_imp = pd.Series(kbest.scores_, index=X.columns)
    feat_imp.sort_values(inplace=True)
    ax = feat_imp.tail(top).plot(kind='barh', figsize=(10,7), title='Feature importance (f_classif)')
    return feat_imp
    

def l1_best_features(X, y, top = 30):
    pass


def rfe_best_features(X, y, top = 30):
    clf = LinearRegression(n_jobs = 6)
    rfe = RFECV(clf, step=1, cv=StratifiedKFold(y, 2), scoring='roc_auc')
    rfe.fit(X, y)
    feat_imp1 = pd.Series(rfe.grid_scores_, index = X.columns)
    
    clf = LogisticRegression(n_jobs = 6)
    rfe = RFECV(clf, step=1, cv=StratifiedKFold(y, 2), scoring='roc_auc')
    rfe.fit(X, y)
    feat_imp2 = pd.Series(rfe.grid_scores_, index = X.columns)
    
    feat_imp11 = pd.concat([feat_imp1, pd.Series(["Linear" for _ in range(len(feat_imp1))])], axis = 1)
    feat_imp22 = pd.concat([feat_imp2, pd.Series(["Logistic" for _ in range(len(feat_imp2))])], axis = 1)
    feat_imp = pd.concat([feat_imp11, feat_imp22])
    return feat_imp1, feat_imp2
    
    plt.figure()
    plt.title("RFE CV (optimal #features = ", rfe.n_features, ")")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.show()
    

def take_features(feat_imp, *args):
    res = []
    for i in range(len(args)):
        res.append(args[i][feat_imp.axes[0]])
    return res
    

def print_cv_scores(clf, X, y, cv = 5, scoring = "roc_auc"):
    print(clf)
    score = cross_validation.cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    print("CV scores: ", score, '\n=>\t', score.mean(), ' (+-', score.std(),')', sep = '')

    
def print_class_report(y_true, y_pred):
    print('Metrics:')
    print(metrics.classification_report(y_true, y_pred))

    
def print_conf_matrix(y_true, y_pred):
    print("Confusion matrix:")
    print("Cls", 0, 1, sep = "\t")
    mat = metrics.confusion_matrix(y_true, y_pred)
    print('0', mat[0][0], mat[0][1], sep = '\t')
    print('1', mat[1][0], mat[1][1], sep = '\t')

    
def per_clf_report(clf, X, y, val_col, cv = 5, scoring = 'roc_auc'):
    y_pred = clf.predict(X[val_col == 1])
    print_cv_scores(clf, X, y, cv, scoring)
    print()
    print_class_report(y[val_col == 1], y_pred)
    print_conf_matrix(y[val_col == 1], y_pred)
    

def print_corr_matrix(clfs, X):
    res = numpy.vstack([x[1].predict_proba(X)[:, 1] for x in clfs])
    mat = numpy.corrcoef(res)
    print("clf", end = "\t")
    print("\t".join([x[0] for x in clfs]))
    for i in range(len(clfs)):
        print(clfs[i][0], end = "\t")
        print("\t".join(map(lambda x: str(round(x, 3)), mat[i, :])))
    plt.pcolor(mat, cmap = plt.cm.RdBu)
    plt.show()


def model_eval_report(clfs, X, y, val_col, cv = 5, scoring = "roc_auc"):
    if not (type(clfs) is list):
        clfs = [clfs]
        
    print("=====================================")
    print("Report table:")
    print("\t".join(["clf", "class", "precis.", "recall", "f1", "support"]))
    print('-------------------------------------')
    for label, clf in clfs:
        clf.fit(X[val_col == 0], y[val_col == 0])
        prec, rec, fscore, supp = metrics.precision_recall_fscore_support(y[val_col == 1], clf.predict(X[val_col == 1]))
        print(label, 0, round(prec[0], 3), round(rec[0], 3), round(fscore[0], 3), supp[0], sep = "\t")
        print("", 1, round(prec[1], 3), round(rec[1], 3), round(fscore[1], 3), supp[1], sep = "\t")
        print("", "avg", round(prec.mean(), 3), round(rec.mean(), 3), round(fscore.mean(), 3), "", sep = "\t")
        print()
    print("=====================================")
    
    print("Correlation matrix:")
    print_corr_matrix(clfs, X[val_col == 0])
    
    print("=====================================")
    
    for label, clf in clfs:
        per_clf_report(clf, X, y, val_col, cv, scoring)
        print("=====================================")
        

def averaging(clfs, X, y, val_col):
    for label, clf in clfs:
        clf.fit(X[val_col == 0], y[val_col == 0])
        
    res = numpy.vstack([x[1].predict_proba(X[val_col == 1])[:, 1] for x in clfs])
    y_pred1 = res.mean(axis=0)
    print("Score (amean):\t", metrics.roc_auc_score(y[val_col == 1], y_pred1))
    y_pred2 = stats.gmean(res, 0)
    print("Score (gmean):\t", metrics.roc_auc_score(y[val_col == 1], y_pred2))
    
    
def train_xgb(train_X, train_y, cv_fold):
    
    def _grid_search(parameters, msg):
        print(msg, end = '...')
        clf_best = GridSearchCV(clf, parameters, cv=cv_fold, n_jobs = 1, scoring = "f1")
        clf_best.fit(train_X, train_y)
        print(' Done.')
        return clf_best.best_estimator_
    
    
    # choose n estimators
    # choose max depth, min child weight
    # again choose n estimators
    # choose gamma
    # tune subsample and colsample by tree
    # tune alpha / lambda
    # reduce learning rate + add more trees
    
    clf = xgb.XGBClassifier(nthread = 7, learning_rate = .05)
    
    parameters = {
        "n_estimators": [100, 200, 300, 600]
    }
    clf = _grid_search(parameters, "1. Number of estimators")
    
    parameters = {
        "scale_pos_weight": [.5, 1, 3, 7, 10, 12, 15]
    }
    clf = _grid_search(parameters, "2. Weight scaling")
    
    parameters = {
        "max_depth": list(range(3, 11, 1)),
        "min_child_weight": list(range(1, 10, 1))
    }
    clf = _grid_search(parameters, "3. Depth and child weights")
    
    parameters = {
        "n_estimators": [100, 200, 300, 600, 800, 1000],
    }
    clf = _grid_search(parameters, "4. Number of estimators")
    
    parameters = { 
        'gamma': [.1*x for x in range(1, 11)]
    }
    clf = _grid_search(parameters, "5. Gamma")
    
    parameters = { 
        'subsample': [.1*x for x in range(4, 11)],
#         'colsample_bytree': [.1*x for x in range(4, 11)]
    }
    clf = _grid_search(parameters, "6. Sample and colsample")
    
    parameters = {
        'reg_alpha':  [1e-5, 1e-3, 1e-2, 0.1, 1, 10, 100],
        'reg_lambda': [1e-5, 1e-3, 1e-2, 0.1, 1, 10, 100]
    }
    clf = _grid_search(parameters, "7. Regularization")
    
    parameters = {
        "n_estimators": [600, 800, 1000, 1300],
        "learning_rate": [.01, .02, .05]
    }
    clf = _grid_search(parameters, "8. Number of estimators and the learning rate")
    
    parameters = {
        "scale_pos_weight": [.5, 1, 3, 7, 10, 12, 15]
    }
    clf = _grid_search(parameters, "9. Weight scaling")
    
    print_cv_scores(clf, train_X, train_y, cv_fold)
    return clf