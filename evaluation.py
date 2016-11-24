"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.1
2016/11/24 05:39PM
"""
from general import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from scipy import interp

import numpy as np
import time

def classifier_list():
    clfs = {}
    ### Random Forests
    clfs['rf'] = RandomForestClassifier(n_jobs=4)
    ### SVM
    clfs['svm'] = SVC(probability=True)
    ### Naive Bayes
    clfs['gnb'] = GaussianNB()
    clfs['bnb'] = BernoulliNB() # Best ?
    clfs['mnb'] = MultinomialNB()
    ### Decision Tree (CART)
    clfs['tree'] = DecisionTreeClassifier()
    return clfs

"""
X: training dataset (features)
y: testing dataset  (label)
clf: classifier
"""
def evaluation(X, y, clf, k_fold=5):
    output = {}
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    mean_precision = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0
    mean_acc = 0.0

    total_ytrue = sum(y)

    cv = StratifiedKFold(n_splits=k_fold)

    i = 0
    train_time = 0
    for (train, test) in cv.split(X, y):
        query_time = time.time()
        fit = clf.fit(X[train], y[train])
        train_time += (time.time() - query_time)
        probas_ = fit.predict_proba(X[test])
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

        # mean_precision += precision
        # mean_recall += recall
        # mean_f1 += f1
        mean_acc += acc
        # debug(roc_auc)
        i += 1
    # mean_tpr /= cv.get_n_splits(X, y)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)

    n_split = cv.get_n_splits(X, y)
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