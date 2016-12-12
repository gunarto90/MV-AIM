"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.6
2016/12/08 04:30PM
"""
"""

"""
from general import *
from view_soft import extract_app_statistics, select_top_k_apps

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import LeaveOneGroupOut
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
import random

MODEL_FILENAME = '{}_{}_{}_{}_{}_{}.bin'  # uid clf_name #iter #total mode

def classifier_list():
    clfs = {}
    ### Forests
    clfs['grf']     = RandomForestClassifier(n_jobs=4, criterion='gini')
    clfs['erf']     = RandomForestClassifier(n_jobs=4, criterion='entropy')
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
    # ### Decision Tree (CART)
    clfs['gdt']     = DecisionTreeClassifier(criterion='gini')
    clfs['edt']     = DecisionTreeClassifier(criterion='entropy')
    clfs['egt']     = ExtraTreeClassifier(criterion='gini')
    clfs['eet']     = ExtraTreeClassifier(criterion='entropy')
    return clfs

def get_cv(k_fold, groups, X, y):
    if groups is None:
        ### Personal CV
        skf  = StratifiedKFold(n_splits=k_fold)
        n_split = skf.get_n_splits(X, y)
        cv = skf.split(X, y)
    else:
        ### Group (leave one subject out)
        logo = LeaveOneGroupOut()
        n_split = logo.get_n_splits(X, y, groups)
        cv = logo.split(X, y, groups=groups)
    return cv, n_split

"""
X: training dataset (features)
y: testing dataset  (label)
clf: classifier
"""
def evaluation(X, y, clf, k_fold=5, info={}, cached=False, mode='Default', groups=None, time_window=0):
    output = {}
    i = 0
    train_time = 0.0
    test_time = 0.0
    mean_acc = 0.0
    total_ytrue = sum(y)

    cv, n_split = get_cv(k_fold, groups, X, y)

    debug('Cross validation : {} times'.format(n_split))
    for (train, test) in cv:
        uid = info.get('uid')
        clf_name = info.get('clf_name')
        filename = None
        if uid is not None and clf_name is not None:
            filename = MODEL_FILENAME.format(uid, clf_name, i, n_split, mode, time_window)

        success = True
        load = False
        query_time = time.time()
        if cached:
            try:
                if filename is None:
                    raise Exception('Filename is None')
                with open(cd.soft_classifier + filename, 'rb') as f:
                    fit = pickle.load(f)
                    load = True
            except:
                success = False
        if not cached or not success:
            fit = clf.fit(X[train], y[train])
        train_time += (time.time() - query_time)
        # probas_ = fit.predict_proba(X[test])
        query_time = time.time()
        inference = fit.predict(X[test])
        test_time += (time.time() - query_time)
        acc = accuracy_score(y[test], inference)

        try:
            if filename is not None and not load:
                with open(cd.soft_classifier + filename, 'wb') as f:
                    pickle.dump(fit, f)
                    debug('Writing to {}'.format(cd.soft_classifier + filename))
        except Exception as ex:
            debug(ex, get_function_name())
        mean_acc += acc
        i += 1

    mean_acc /= n_split
    train_time /= n_split
    test_time /= n_split

    output['acc']           = mean_acc
    output['time_train']    = train_time
    output['time_test']     = test_time

    return output

"""
Application view's evaluation using top-k and scoring method
"""
def test_soft(top_k_apps, X, y, mode, app_names, categories, weight_mode, topk):
    ncol = X.shape[1]
    correct = 0.0
    for i in range(len(X)):
        temp = []
        scores = np.zeros(len(var.activities))
        ## Transform binary values into name list
        # debug(sum(X[i]))
        for j in range(ncol):
            if X[i][j] == 1:
                if mode.lower() == 'full' or mode.lower() == 'part':
                    name = app_names[j]
                elif mode.lower() == 'cat':
                    name = categories[j]
                temp.append(name)
        # debug(temp)
        ## Evaluate on each
        for j in range(len(var.activities)):
            top = top_k_apps[j]
            if len(top) > 0:
                for name in temp:
                    found = top.get(name)
                    if found is not None:
                        (value, rank) = found
                        if weight_mode == 'g':
                            scores[j] += 1.0
                        elif weight_mode == 'w':
                            scores[j] += len(top) - rank
                        elif weight_mode == 'f':
                            scores[j] += value['f']
                        elif weight_mode == 'e':
                            scores[j] += (1 - value['e'])
                        elif weight_mode == 'ef':
                            e = 1 - value['e']
                            f = value['f']
                            scores[j] += (e*f)
                        elif weight_mode == 'erf':
                            e = 1 - value['e']
                            f = sqrt(value['f'])
                            scores[j] += (e*f)
        ## Get the "max" scores
        # debug(scores)
        max_score = max(scores)
        inferences = []
        for i in range(len(var.activities)):
            if scores[i] == max_score:
                inferences.append(i)
        inference = random.choice(inferences)
        # print(inference, y[i])
        if inference == y[i]:
            correct += 1
    correct /= len(X)
    return correct

def soft_evaluation(data, uid, mode, topk, sorting, sort_mode, weight_mode, app_names=None, categories=None, cached=True, k_fold=5, groups=None):
    data = np.array(data)
    ncol = data.shape[1]
    X = data[:,3:ncol] # Remove index 0 (uid), index 1 (time), and index 2 (activities)
    y = data[:,2]

    train_time = 0.0
    test_time = 0.0
    mean_acc = 0.0

    cv, n_split = get_cv(k_fold, groups, X, y)
    counter = 0
    output = {}
    for (train, test) in cv:
        ### Training
        query_time = time.time()
        acts_app = extract_app_statistics(X[train], y[train], mode, uid, sort_mode, weight_mode, app_names=app_names, categories=categories, cached=cached, counter=counter, length=n_split)
        top_k_apps = select_top_k_apps(topk, acts_app, sorting, sort_mode)
        train_time += (time.time() - query_time)
        ### Testing
        query_time = time.time()
        mean_acc += test_soft(top_k_apps, X, y, mode, app_names, categories, weight_mode, topk)
        test_time += (time.time() - query_time)
        ### Print top-k apps
        # for i in range(len(var.activities)):
        #     top = top_k_apps[i]
        #     if len(top) > 0:
        #         debug(var.activities[i], clean=True)
        #         debug('app,entropy,frequency', clean=True)
        #         for app_id, (value, k) in sorted(top.items(), key=lambda value: value[1][1]):   # Sort using rank
        #             debug('{} : [e:{}, f:{}, rank:{}]'.format(app_id, value['e'], value['f'], k), clean=True)
        #         print()
        ### Increment counter
        counter += 1

    mean_acc /= n_split
    train_time /= n_split
    test_time /= n_split

    output['acc']           = mean_acc
    output['time_train']    = train_time
    output['time_test']     = test_time

    debug(output)

    return output