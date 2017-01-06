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

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, KFold

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz
from sklearn.decomposition import PCA, SparsePCA

from scipy import interp
from math import sqrt

import numpy as np
import time
import scipy
import pickle
import random
import gc
import psutil

gc.enable()

MODEL_FILENAME  = '{}_{}_{}_{}_{}_{}_{}_{}{}{}.bin'   # [uid] [clf_name] [#iter] [#total] [mode] [TIME_WINDOW] [PCA] [TimeInfo] [(None)/fore/back] [(None)/over/under/combo]

DUMP_XY         = False
DUMPXY_FILENAME = '{}_{}_{}_{}_{}_{}_{}_{}.txt'     # [uid] [clf_name] [#iter] [#total] [mode] [TIME_WINDOW] [PCA] [TimeInfo]

def classifier_list():
    clfs = {}
    ### Forests
    clfs['rfg']     = RandomForestClassifier(n_jobs=4, criterion='gini')
    clfs['rfe']     = RandomForestClassifier(n_jobs=4, criterion='entropy')
    # clfs['etr']     = ExtraTreesClassifier()
    ### Boosting
    # clfs['gbc']     = GradientBoostingClassifier()
    # clfs['ada']     = AdaBoostClassifier()
    # clfs['bag']     = BaggingClassifier()
    ### SVM
    # clfs['lsvm']    = LinearSVC()
    # clfs['qsvm']    = SVC(probability=True, kernel='poly', degree=2)  # Slow
    # clfs['psvm']    = SVC(probability=True, kernel='poly', degree=3)  # Slow
    # clfs['ssvm']    = SVC(probability=True, kernel='sigmoid')         # Slow
    # clfs['rsvm']    = SVC(probability=True, kernel='rbf')             # Slow
    ### Naive Bayes
    # clfs['gnb']     = GaussianNB()      # Worst
    # clfs['nbb']     = BernoulliNB()     # Good
    # clfs['nbm']     = MultinomialNB()   # Best    # Can't handle negatives
    # ### Decision Tree (CART)
    clfs['dtg']     = DecisionTreeClassifier(criterion='gini')
    clfs['dte']     = DecisionTreeClassifier(criterion='entropy')
    # clfs['etg']     = ExtraTreeClassifier(criterion='gini')
    # clfs['ete']     = ExtraTreeClassifier(criterion='entropy')
    return clfs

def get_cv(k_fold, groups, X, y):
    if groups is None:
        ### Personal CV
        skf  = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=int(len(X)/2))
        # skf  = KFold(n_splits=k_fold, shuffle=True, random_state=int(len(X)/2))
        n_split = skf.get_n_splits(X, y)
        cv = skf.split(X, y)
    else:
        ### Group (leave one subject out)
        logo = LeaveOneGroupOut()
        n_split = logo.get_n_splits(X, y, groups)
        cv = logo.split(X, y, groups=groups)
    return cv, n_split

def matrix_decomposition(X, y, sparse=False):
    n_components = 20
    svd_solver = 'full'
    # svd_solver = 'randomized'     # A randomized algorithm for the decomposition of matrices
    if sparse:
        pca = SparsePCA(n_components=n_components, svd_solver=svd_solver)
    else:
        # M. Tipping and C. Bishop, Probabilistic Principal Component Analysis, Journal of the Royal Statistical Society, Series B, 61, Part 3, pp. 611-622 (Default)
        pca = PCA(n_components=n_components, svd_solver=svd_solver)
        # pca = PCA(n_components='mle', svd_solver='full')
        # Thomas P. Minka: Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
    return pca.fit_transform(X, y)

def sampling(X, y, method='over'):
    debug('Started sampling')

    ### ovesampling
    if method== 'over':
        query_time = time.time()
        pp = SMOTE(kind='regular')
        X_pp, y_pp = pp.fit_sample(X, y)
        process_time = int(time.time() - query_time)
        debug('Finished sampling SMOTE in {} seconds'.format(process_time))

    ### undersampling
    elif method == 'under':
        query_time = time.time()
        pp = RandomUnderSampler()
        X_pp, y_pp = pp.fit_sample(X, y)
        process_time = int(time.time() - query_time)
        debug('Finished sampling One-Sided Selection in {} seconds'.format(process_time))
    
    ### oversampling + undersampling
    elif method =='combo':
        query_time = time.time()
        pp = SMOTEENN()
        X_pp, y_pp = pp.fit_sample(X, y)
        process_time = int(time.time() - query_time)
        debug('Finished sampling SMOTE-ENN in {} seconds'.format(process_time))
    
    return X_pp, y_pp

"""
X: training dataset (features)
y: testing dataset  (label)
clf: classifier
"""
def evaluation(X, y, clf, k_fold=5, info={}, cached=False, mode='Default', groups=None, time_window=0, pca=False, time_info=False, app_type='all', method='original'):
    output = {}
    i = 0
    train_time = 0.0
    test_time = 0.0
    mean_acc = 0.0
    total_ytrue = sum(y)

    str_pca = ''
    str_timeinfo = ''
    str_app_type = ''
    str_pre = ''

    if pca:
        str_pca = 'PCA'
        X = matrix_decomposition(X, y, False)
    if time_info:
        str_timeinfo = 'TimeInfo'
    if app_type != 'all':
        str_app_type = app_type
    if method != 'original':
        str_pre = '_' + method

    # debug(X)
    # debug(y)

    cv, n_split = get_cv(k_fold, groups, X, y)

    debug('Cross validation : {} times'.format(n_split))
    for (train, test) in cv:
        # debug((X[train]))
        # debug((X[test]))
        if method != 'original':
            Xt, yt = sampling(X[train], y[train], method=method)
        else:
            Xt, yt = X[train], y[train]
        uid = info.get('uid')
        clf_name = info.get('clf_name')
        filename = None
        if uid is not None and clf_name is not None:
            filename = MODEL_FILENAME.format(uid, clf_name, i, n_split, mode, time_window, str_pca, str_timeinfo, str_app_type, str_pre)

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
        try:
            if not cached or not success:
                fit = clf.fit(Xt, yt)
        except Exception as ex:
            debug(ex, get_function_name())
        train_time += (time.time() - query_time)
        if success:
            debug('Model loaded from {}'.format(filename))
        # probas_ = fit.predict_proba(X[test])
        query_time = time.time()
        inference = fit.predict(X[test])
        test_time += (time.time() - query_time)
        acc = accuracy_score(y[test], inference)

        if clf_name == 'dte' or clf_name == 'dtg':
            try:
                export_graphviz(clf, out_file=cd.soft_classifier + filename + '.dot')
                # str_tree = export_graphviz(clf, out_file=None)
                # debug(str_tree)
            except Exception as ex:
                debug(ex, get_function_name())

        try:
            if filename is not None and not load:
                with open(cd.soft_classifier + filename, 'wb') as f:
                    pickle.dump(fit, f)
                    debug('Writing to {}'.format(cd.soft_classifier + filename))
        except Exception as ex:
            debug(ex, get_function_name())
        mean_acc += acc
        debug('[{}] [{}] Accuracy [{} of {}] : {}'.format(uid, clf_name, i, n_split, acc))
        ### Clear memory
        inference = None
        gc.collect()
        debug(psutil.virtual_memory())
        ### Dump dataset
        # if DUMP_XY:
        #     z = np.array([y])
        #     filename = cd.soft_classifier + DUMPXY_FILENAME.format(uid, clf_name, i, n_split, mode, time_window, str_pca, str_timeinfo)
        #     stack = np.concatenate((X, z.T), axis=1)
        #     np.savetxt(filename, stack, delimiter=',')
        #     stack = None
        ### Increment counter
        i += 1

    ## Clear memory
    fit = None

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
def test_soft(app_models, X, y, mode, app_names, categories, weight_mode, topk):
    ncol = X.shape[1]
    correct = 0.0
    if mode.lower() == 'cat':
        names = list(categories.values())
    else:
        names = list(app_names)
    for i in range(len(X)):
        scores = []
        ### Name in X (for debugging)
        # catch = []
        # for j in range(len(X[i])):
        #     if X[i][j] > 0:
        #         catch.append(names[j])
        ### Calculate score for each activity
        for j in range(len(var.activities)):
            score = np.multiply(X[i], app_models[j])
            scores.append(sum(score))
        max_score = max(scores)
        inferences = []
        for j in range(len(var.activities)):
            if scores[j] == max_score:
                inferences.append(j)
        inference = random.choice(inferences)

        if inference == y[i]:
            correct += 1
        # else:
        #     print(inference, y[i], inferences, scores, sum(X[i]), catch)
    correct /= len(X)
    return correct

def soft_evaluation(data, uid, mode, topk, sorting, sort_mode, weight_mode, app_names=None, categories=None, cached=True, k_fold=5, groups=None):
    data = np.array(data)
    ncol = data.shape[1]
    base_col = 3
    # if time_info:
    #     base_col += 3
    X = data[:,base_col:ncol] # Remove index 0 (uid), index 1 (time), and index 2 (activities)
    y = data[:,2]

    # debug(data)
    # debug(X)
    # debug(y)

    # debug(X.shape)

    train_time = 0.0
    test_time = 0.0
    mean_acc = 0.0

    cv, n_split = get_cv(k_fold, groups, X, y)
    counter = 0
    output = {}
    if mode.lower() == 'cat' or mode.lower() == 'hybrid':
        names = list(categories.values())
    else:
        names = list(app_names)
    # debug(names)
    for (train, test) in cv:
        ### Training
        query_time = time.time()
        acts_app = extract_app_statistics(X[train], y[train], mode, uid, sort_mode, weight_mode, app_names=app_names, categories=categories, cached=cached, counter=counter, length=n_split)
        top_k_apps = select_top_k_apps(topk, acts_app, sorting, sort_mode)
        app_models = []
        for i in range(len(var.activities)):
            temp = []
            app_models.append(temp)
            for j in range(len(names)):
                temp.append(0.0)
            top = top_k_apps[i]
            if len(top) > 0:
                for app_id, (value, rank) in top.items():
                    # idx = names.index(app_id)
                    idx = app_id
                    if idx != -1:
                        if weight_mode == 'g':
                            score = 1.0
                        elif weight_mode == 'w':
                            score = len(top) - rank
                        elif weight_mode == 'f':
                            score = value['f']
                        elif weight_mode == 'e':
                            score = (1 - value['e'])
                        elif weight_mode == 'ef':
                            e = 1 - value['e']
                            f = value['f']
                            score = (e*f)
                        elif weight_mode == 'erf':
                            e = 1 - value['e']
                            f = sqrt(value['f'])
                            score= (e*f)
                        elif weight_mode == 'we':
                            e = 1 - value['e']
                            w = len(top) - rank
                            score = (e*w)
                        elif weight_mode == 'wef':
                            e = 1 - value['e']
                            w = len(top) - rank
                            f = value['f']
                            score = (e*w*f)
                        elif weight_mode == 'werf':
                            e = 1 - value['e']
                            w = len(top) - rank
                            f = sqrt(value['f'])
                            score = (e*w*f)
                        temp[idx] = score
        train_time += (time.time() - query_time)
        ### Testing
        query_time = time.time()
        mean_acc += test_soft(app_models, X[test], y[test], mode, app_names, categories, weight_mode, topk)
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

def soft_time_evaluation():
    pass