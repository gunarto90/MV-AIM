"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.4
2016/11/29 11:23AM
"""
from general import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from scipy import interp
from math import sqrt

import numpy as np
import time
import scipy
import pickle

MODEL_FILENAME = '{}_{}_{}_{}_{}.bin'  # uid clf_name #iter #total mode

def classifier_list():
    clfs = {}
    ### Forests
    clfs['grf']     = RandomForestClassifier(n_jobs=4, criterion='gini')
    clfs['erf']     = RandomForestClassifier(n_jobs=4, criterion='entropy')
    # clfs['isf']     = IsolationForest()
    clfs['etr']     = ExtraTreesClassifier()
    ### Boosting
    clfs['gbc']     = GradientBoostingClassifier()
    clfs['ada']     = AdaBoostClassifier()
    clfs['bag']     = BaggingClassifier()
    ### SVM
    clfs['lsvm']    = LinearSVC()
    # clfs['qsvm']    = SVC(probability=True, kernel='poly', degree=2)  # Slow
    # clfs['psvm']    = SVC(probability=True, kernel='poly', degree=3)  # Slow
    # clfs['ssvm']    = SVC(probability=True, kernel='sigmoid')         # Slow
    # clfs['rsvm']    = SVC(probability=True, kernel='rbf')             # Slow
    ### Naive Bayes
    clfs['gnb']     = GaussianNB()      # Worst
    clfs['bnb']     = BernoulliNB()     # Good
    clfs['mnb']     = MultinomialNB()   # Best
    ### Decision Tree (CART)
    clfs['gdt']     = DecisionTreeClassifier(criterion='gini')
    clfs['edt']     = DecisionTreeClassifier(criterion='entropy')
    clfs['egt']     = ExtraTreeClassifier(criterion='gini')
    clfs['eet']     = ExtraTreeClassifier(criterion='entropy')
    return clfs

"""
X: training dataset (features)
y: testing dataset  (label)
clf: classifier
"""
def evaluation(X, y, clf, k_fold=5, info={}, cached=False, mode='Default'):
    output = {}
    # mean_tpr = 0.0
    # mean_fpr = np.linspace(0, 1, 100)

    # mean_precision = 0.0
    # mean_recall = 0.0
    # mean_f1 = 0.0
    mean_acc = 0.0

    total_ytrue = sum(y)

    cv = StratifiedKFold(n_splits=k_fold)

    i = 0
    train_time = 0
    n_split = cv.get_n_splits(X, y)
    for (train, test) in cv.split(X, y):
        uid = info.get('uid')
        clf_name = info.get('clf_name')
        filename = None
        if uid is not None and clf_name is not None:
            filename = MODEL_FILENAME.format(uid, clf_name, i, n_split, mode)

        query_time = time.time()
        if cached:
            try:
                if filename is None:
                    raise Exception('Filename is None')
                with open(cd.model_folder + filename, 'rb') as f:
                    fit = pickle.load(f)
            except:
                fit = clf.fit(X[train], y[train])
        elif cached is False:
            fit = clf.fit(X[train], y[train])
        train_time += (time.time() - query_time)
        # probas_ = fit.predict_proba(X[test])
        inference = fit.predict(X[test])
        # Compute ROC curve and area the curve
        # fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        # mean_tpr += interp(mean_fpr, fpr, tpr)
        # mean_tpr[0] = 0.0
        # roc_auc = auc(fpr, tpr)

        # precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1])
        average = 'weighted'
        # precision = precision_score(y[test], inference, average=average)
        # recall = recall_score(y[test], inference, average=average)
        # f1 = f1_score(y[test], inference, average=average)
        acc = accuracy_score(y[test], inference)

        if filename is not None:
            with open(cd.model_folder + filename, 'wb') as f:
                pickle.dump(fit, f)

        # mean_precision += precision
        # mean_recall += recall
        # mean_f1 += f1
        mean_acc += acc
        # debug(roc_auc)
        i += 1
    # mean_tpr /= n_split
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)

    # mean_precision /= n_split
    # mean_recall /= n_split
    # mean_f1 /= n_split
    mean_acc /= n_split
    train_time /= n_split

    # output['auc']   = mean_auc
    # output['p']     = mean_precision
    # output['r']     = mean_recall
    # output['f1']    = mean_f1
    output['acc']   = mean_acc
    output['time']   = train_time
    # output['y1']    = total_ytrue

    return output