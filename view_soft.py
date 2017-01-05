"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.12
2016/12/08 04:34PM
"""

"""
## ToDo !!:
"""

"""

"""
import getopt
import sys
import re
import os
import json
import pickle
import operator
import psutil
import gc
import random

import numpy as np
np.seterr(divide='ignore', invalid='ignore')    ### Remove warning from divide by zero and nan
import config_directory as cd
import config_variable as var

from datetime import datetime, date
from string import digits
from collections import defaultdict
from math import sqrt

from general import *
from evaluation import *

gc.enable()
np.set_printoptions(precision=3, suppress=True)

### Supporting files
CATEGORY_NAME           = 'category_lookup.csv'
APP_CATEGORY            = 'app_category.csv'
STOP_FILENAME           = 'stop_app.txt'
# APP_NAME_LIST           = 'app_{}.csv'
APP_NAME_LIST           = 'app_{}_{}.csv' ## [full/cat/part/hybrid] [fore/back/all]

### Intermediate binary file
USERS_DATA_NAME         = 'users_data_{}_{}_{}{}{}.bin'                         ## [Mode] [UID] [Time Window] [(None)/_Time] [None/_(fore/back)]
APP_STATS_NAME          = 'acts_stats_{}_{}_SORT[{}]_WEIGHT[{}]_{}_{}.bin'      ## [Mode] [UID] [Sorting Mode] [Weighting Mode] [Counter] [Length]

### Software dataset and intermediate
SOFT_FORMAT             = '{}/{}_soft.csv'                                      ## Original app data    [Directory] [UID]
SOFT_PROCESSED          = '{}/{}_soft_{}_{}.csv'                                ## Processed: part name [Directory] [UID] [Mode] [Time Window]

### Reports
REPORT_NAME             = '{}/soft_report_{}_{}_{}_{}_{}_{}.csv'                ## [Directory] [agg/single] [Mode] [Time Window] [PCA] [TimeInfo] [Today]
REPORT_TOPK_NAME        = 'soft_report_topk_{}_{}_{}_{}{}.csv'                  ## [single/agg] [full/part/cat] [Time Window] [(None)/_Time] [Today]

REPORT_TOD_NAME         = '{}/soft_report_tod_{}_{}.csv'                        ## [Directory] [Mode] [UID]
REPORT_DOW_NAME         = '{}/soft_report_dow_{}_{}.csv'                        ## [Directory] [Mode] [UID]
REPORT_TOW_NAME         = '{}/soft_report_tow_{}_{}.csv'                        ## [Directory] [Mode] [UID]

### Time data
TIME_CACHE              = 'time_data_{}_{}_{}_{}.bin'                           ## [tod/dow/tow/atod/adow/atow] [UID] [iteration] [n_split]

APP_F_THRESHOLD         = 1000  ## Minimum occurrence of the app throughout the dataset

K_FOLD                  = 5

DEFAULT_SORTING         = 'f'       ### ef: entropy frequency, f: frequency, erf: entropy*sqrt(freqency)
DEFAULT_WEIGHTING       = 'g'       ### g: general (unweighted), w: rank on list, f: frequency, e: 1-entropy, ef: e*f, erf: e*sqrt(f)
DEFAULT_TIME_WINDOW     = 1000      ### in ms

COLUMN_NAMES            = 'UID,Classifier,Accuracy,TrainTime(s),TestTime(s),Mode,TimeWindow,PCA,TimeInfo,AppType'

WRITE_HEADER            = True

"""
@Initialization methods
"""
def init_stop_words(stop_app_filename):
    stop_words = []
    with open(stop_app_filename, 'r') as fr:
        for line in fr:
            stop_words.append(line.strip())
    return stop_words

def init_app_category(mode, app_type):
    filename = cd.software_folder + APP_NAME_LIST.format(mode, app_type)
    app_cat = {}
    categories = {}
    with open(filename) as f:
        for line in f:
            split = line.strip().split(',')
            app_name = split[0]
            cat_id = int(split[1])
            cat_name = split[2]
            if mode.lower() == 'hybrid' and cat_name == '#Internal':
                categories[cat_id] = app_name
            else:
                categories[cat_id] = cat_name
            app_cat[app_name] = cat_id
    # for app_name, cat_id in app_cat.items():
    #     debug('{},{}'.format(app_name, cat_id), clean=True)
    return categories, app_cat

def init_sorting_schemes():
    sorting = {}
    ### Sorting mechanism, reverse (descending order) status [True/False]
    sorting['erf']  = (lambda value: (1-value[1]['e'])*sqrt(value[1]['f']), True)
    sorting['ef']   = (lambda value: (1-value[1]['e'])*value[1]['f'], True)
    sorting['f']    = (lambda value: value[1]['f'], True)
    sorting['e']    = (lambda value: value[1]['e'], False)
    return sorting

"""
Generating dataset from raw file for the first time
"""
def transform_dataset(user_ids, app_names, mode, write=False, categories=None, app_cat=None, cached=False, time_window=0, time_info=False, app_type='all'):
    ### If not cached
    remove_digits = str.maketrans('', '', digits)
    users_data = {}
    ctr_uid = 0
    str_time_info = ''
    str_app_type = ''
    if time_info:
        str_time_info = '_Time'
    if app_type != 'all':
        str_app_type = '_' + app_type
    for uid in user_ids:
        ctr_uid += 1
        lines = []
        filename = SOFT_FORMAT.format(cd.dataset_folder, uid)
        debug('Transforming : {} [{}/{}]'.format(filename, ctr_uid, len(user_ids)), callerid=get_function_name(), out_file=True)
        ### Load cache
        user_data = None
        binary_filename = USERS_DATA_NAME.format(mode, uid, time_window, str_time_info, str_app_type)
        if cached:
            try:
                with open(cd.soft_users_cache + binary_filename, 'rb') as f:
                    user_data = pickle.load(f)
                    users_data[uid] = user_data
                    debug('Cache file loaded: {}'.format(cd.soft_users_cache + binary_filename))
            except Exception as ex:
                # debug(ex, get_function_name())
                debug('Cache file not found: {}'.format(cd.soft_users_cache + binary_filename))
        if user_data is None:
            ctr = 0
            num_lines = sum(1 for line in open(filename))
            user_data = []
            users_data[uid] = user_data
            with open(filename) as fr:
                time = 0
                previous_time = 0
                app_dist = []
                if mode.lower() == 'full' or mode.lower() == 'part':
                    for i in range(len(app_names)):
                        app_dist.append(False)
                elif mode.lower() == 'cat' or mode.lower() == 'hybrid':
                    for i in range(len(categories)):
                        app_dist.append(False)
                for line in fr:
                    split = line.lower().strip().split(',')
                    uid = int(split[0])
                    act = split[1]
                    app = split[2]
                    time = int(split[3])    # in ms
                    act_int = activity_to_int(act, var.activities)
                    if act_int < 0:
                        continue
                    # print(act_int)
                    date = datetime.fromtimestamp(time / 1e3)
                    if mode.lower() == 'cat' or mode.lower() == 'hybrid':
                        cat = app_cat.get(app.strip())
                        if cat is not None:
                            try:
                                if app_type == 'fore':
                                    cat -= 1
                                app_dist[cat] = True
                            except Exception as ex:
                                debug('Category: {}, len(app_dist): {}, cat: {}'.format(cat, len(app_dist), app.strip()))
                                debug(ex, get_function_name())
                    elif mode.lower() == 'full':
                        try:
                            idx = app_names.index(app)
                            if idx != -1:
                                app_dist[idx] = True
                        except:
                            ### Because some app names are deleted to save resources
                            pass
                    elif mode.lower() == 'part':
                        app_split = app.translate(remove_digits).replace(':','.').split('.')
                        for app_id in app_split:
                            try:
                                idx = app_names.index(app_id)
                                if idx != -1:
                                    app_dist[idx] = True
                            except:
                                ### Because some app names are deleted to save resources
                                pass
                    if abs(time - previous_time) >= time_window or ctr == num_lines-1:
                        if np.any(app_dist):
                            # soft = (','.join(str(x) for x in app_dist))
                            # text = '{},{},{}'.format(uid, act_int, soft)
                            # lines.append(text)
                            ### label is put in the first column
                            data = []
                            data.append(uid)
                            data.append(time)
                            data.append(act_int)
                            if time_info:
                                date = datetime.fromtimestamp(time / 1e3)
                                day = date.weekday()
                                hour = date.hour
                                timeweek = day*24 + hour
                                data.append(day)
                                data.append(hour)
                                data.append(timeweek)
                            data.extend(app_dist)
                            user_data.append(data)
                            # debug(data)
                            ## Reset app distributions
                            app_dist = []
                            if categories is None:
                                for i in range(len(app_names)):
                                    app_dist.append(False)
                            else:
                                for i in range(len(categories)):
                                    app_dist.append(False)
                    ### Finally update the previous time to match current time
                    previous_time = time

                    ctr += 1
                    # if ctr % 100000 == 0:
                    #     debug('Processing {:,} lines'.format(ctr), out_file=True)
                debug('len(file) : {}'.format(ctr))
            debug('Started writing all app and users data into binary files', out_file=True)
            if write:
                with open(cd.soft_users_cache + binary_filename, 'wb') as f:
                    pickle.dump(user_data, f)
            debug('Finished writing all app and users data into binary files', out_file=True)
    return users_data

"""
Generating all apps names from raw file for the first time
"""
def get_all_apps(user_ids, stop_words, mode, app_type, write=False, cached=False):
    ### Cached
    if mode.lower() == 'full' or mode.lower() == 'part':
        app_filename = cd.software_folder + APP_NAME_LIST.format(mode.lower(), app_type)
    else:
        app_filename = cd.software_folder + APP_NAME_LIST.format('full', app_type)
    app_names = []
    if cached:
        try:
            with open(app_filename) as fr:
                for line in fr:
                    split = line.strip().split(',')
                    try:
                        f = int(split[1])
                        if f > APP_F_THRESHOLD:
                            app_names.append(split[0])
                    except:
                        pass
        except Exception as ex:
            debug(ex, get_function_name())
        if len(app_names) > 0:
            return app_names
    ### If not cached
    remove_digits = str.maketrans('', '', digits)
    app_names = defaultdict(int)
    debug('Starting get all app names')
    for uid in user_ids:
        filename = SOFT_FORMAT.format(cd.dataset_folder, uid)
        try:
            with open(filename) as fr:
                debug(filename, callerid=get_function_name())
                for line in fr:
                    split = line.lower().strip().split(',')
                    app = split[2]
                    if mode.lower() == 'part':
                        app_split = app.translate(remove_digits).replace(':','.').split('.')
                        for app_id in app_split:
                            app_names[app_id] += 1
                    if mode.lower() == 'full':
                        split = app.split(':')
                        app_id = split[0]
                        app_names[app_id] += 1
        except Exception as ex:
            debug('Exception: {}'.format(ex), get_function_name())
    debug(app_names)
    for app in stop_words:
        app_names.pop(app, None)
    debug('Finished get all app names')
    if write:
        remove_file_if_exists(app_filename)
        texts = []
        for k, v in app_names.items():
            texts.append('{},{}'.format(k,v))
        write_to_file_buffered(app_filename, texts)
    return app_names.keys()

### label is put in the first column
def testing(dataset, uid, mode, cached=True, groups=None, time_window=DEFAULT_TIME_WINDOW, pca=False, time_info=False, app_type='all'):
    debug('Processing: {}'.format(uid))
    dataset = np.array(dataset)
    clfs = classifier_list()
    try:
        # debug(dataset.shape)
        # debug(dataset[0])
        ncol = dataset.shape[1]
        base_col = 3
        if time_info:
            base_col += 3
        X = dataset[:,base_col:ncol] # Remove index 0 (uid), index 1 (time), and index 2 (activities)
        y = dataset[:,2]
        texts = []
        info = {}
        info['uid'] = uid
        for name, clf in clfs.items():
            debug(name)
            info['clf_name'] = name
            output = evaluation(X, y, clf, k_fold=K_FOLD, info=info, cached=cached, mode=mode, groups=groups, time_window=time_window, pca=pca, time_info=time_info, app_type=app_type)
            acc = output['acc']
            time_train = output['time_train']
            time_test = output['time_test']
            text = '{},{},{},{},{},{},{},{},{},{}'.format(uid, name, acc, time_train, time_test, mode, time_window, pca, time_info, app_type)
            texts.append(text)
        # Clear memory
        X = None
        y = None
        return texts
    except Exception as ex:
        debug(ex, get_function_name())
        raise Exception(ex)
        return None

### Generating testing report per user
def generate_testing_report(users_data, user_ids, mode, clear_data=False, categories=None, cached=True, agg=False, time_window=DEFAULT_TIME_WINDOW, pca=False, time_info=False, app_type='all'):
    # ### Test
    if not agg:
        debug('Evaluating application data (single)', out_file=True)
    else:
        debug('Evaluating application data (agg)', out_file=True)
    dataset = []
    groups  = []
    output = []
    if WRITE_HEADER:
        output.append(COLUMN_NAMES)
    ctr_uid = 0
    for uid, data in users_data.items():
        ctr_uid += 1
        debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(users_data)), out_file=True)
        debug('#Rows: {}'.format(len(data)), out_file=True)
        debug(psutil.virtual_memory())
        if uid not in user_ids:
            continue
        if not agg:
            result = testing(data, uid, mode=mode, cached=cached, time_window=time_window, pca=pca, time_info=time_info, app_type=app_type)
            if result is not None:
                output.extend(result)
        else:
            for x in data:
                dataset.append(x)
                groups.append(ctr_uid)
    if agg:
        uid = 'ALL'
        result = testing(dataset, uid, mode=mode, cached=cached, groups=groups, time_window=time_window, pca=pca, time_info=time_info, app_type=app_type)
        if result is not None:
            output.extend(result)
    if clear_data:
        try:
            del dataset[:]
        except Exception as ex:
            debug(ex, get_function_name())
    if agg:
        agg_name = 'agg'
    else:
        agg_name = 'single'
    if pca:
        pca_name = 'pca'
    else:
        pca_name = 'normal'
    if time_info:
        time_name = 'timeinfo'
    else:
        time_name = 'notime'
    filename = REPORT_NAME.format(cd.soft_report, agg_name, mode, time_window, pca_name, time_name, date.today())
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, output)
    output = None
    # debug(output)

def init_time_matrix(slots):
    matrix = {}
    for i in range(slots):
        matrix[i] = np.zeros((len(var.activities),))
    return matrix

def build_time_matrix(uid, X, y, columns, n_iter, total, cached=False):
    nameset = ['tod', 'dow', 'tow', 'atod', 'adow', 'atow']
    filenames = {}
    time_app_data = {}

    for name in nameset:
        filenames[name] = TIME_CACHE.format(name, uid, n_iter, total)

    tod_file = TIME_CACHE.format('tod', uid, n_iter, total)
    dow_file = TIME_CACHE.format('dow', uid, n_iter, total)
    tow_file = TIME_CACHE.format('tow', uid, n_iter, total)

    atod_file = TIME_CACHE.format('atod', uid, n_iter, total)
    adow_file = TIME_CACHE.format('adow', uid, n_iter, total)
    atow_file = TIME_CACHE.format('atow', uid, n_iter, total)

    cache_folder = cd.soft_users_time_cache

    if cached:
        success = True
        for name in nameset:
            try:
                with open(cache_folder + filenames[name], 'rb') as f:
                    time_app_data[name] = pickle.load(f)
            except:
                success = False
        if not success:
            build_time_matrix(uid, X, y, columns, n_iter, total, cached=False)            
    else:
        time_app_data['tod'] = init_time_matrix(24)
        time_app_data['dow'] = init_time_matrix(7)        
        time_app_data['tow'] = init_time_matrix(7*24)

        time_app_data['atod'] = {}
        time_app_data['adow'] = {}
        time_app_data['atow'] = {}

        for i in range(1, X.shape[1]):
            time_app_data['atod'][i-1] = init_time_matrix(24)
            time_app_data['adow'][i-1] = init_time_matrix(7)
            time_app_data['atow'][i-1] = init_time_matrix(7*24)

        for i in range(len(X)):
            time = X[i][0]
            act_int = y[i]
            if not time_info:
                date = datetime.fromtimestamp(time / 1e3)
                day = date.weekday()
                hour = date.hour
                timeweek = day*24 + hour
            else:
                day = user_data[3]
                hour = user_data[4]
                timeweek = user_data[5]
            ## Build the activity distribution among all timeslots
            time_app_data['tod'][hour][act_int] += 1
            time_app_data['dow'][day][act_int] += 1
            time_app_data['tow'][timeweek][act_int] += 1
            for j in range(1, len(X[i])):
                if X[i][j] == 1:
                    time_app_data['atod'][j-1][hour][act_int] += 1
                    time_app_data['adow'][j-1][day][act_int] += 1
                    time_app_data['atow'][j-1][timeweek][act_int] += 1

        ## Normalize the value
        for name, data in time_app_data.items():
            if name in ['tod', 'dow', 'tow']:
                for i, arr in data.items():
                    total = float(sum(arr))
                    if total > 0:
                        for j in range(len(arr)):
                            arr[j] = arr[j]/total
            elif name in ['atod', 'adow', 'atow']:
                for x in range(len(columns)):
                    for i, arr in data[x].items():
                        total = float(sum(arr))
                        if total > 0:
                            for j in range(len(arr)):
                                arr[j] = arr[j]/total

        ## Dump into file
        for name in nameset:
            with open(cache_folder + filenames[name], 'wb') as f:
                pickle.dump(time_app_data[name], f)

    return time_app_data

def get_answer(arr):
    max_score = max(arr)
    inferences = []
    for j in range(len(var.activities)):
        if arr[j] == max_score:
            inferences.append(j)
    inference = random.choice(inferences)
    return inference

def testing_time_app(X, y, time_app_data):
    scores = {}
    for name in time_app_data:
        scores[name] = 0.0
    iterator = 0
    for x in X:
        ## Extract time info
        date = datetime.fromtimestamp(x[0] / 1e3)
        day = date.weekday()
        hour = date.hour
        timeweek = day*24 + hour
        # debug(scores['dow'])
        # debug(time_app_data['dow'][day])
        ## Extract score for each method
        tod = time_app_data['tod'][hour]
        dow = time_app_data['dow'][day]
        tow = time_app_data['tow'][timeweek]

        # debug(tod)
        # debug(dow)
        # debug(tow)

        answer_tod = get_answer(tod)
        answer_dow = get_answer(dow)
        answer_tow = get_answer(tow)

        # debug(answer_tod)
        # debug(answer_dow)
        # debug(answer_tow)
        # debug(y[iterator])

        if answer_tod == y[iterator]:
            scores['tod'] += 1
        if answer_dow == y[iterator]:
            scores['dow'] += 1
        if answer_tow == y[iterator]:
            scores['tow'] += 1

        atod = np.zeros((len(var.activities),))
        adow = np.zeros((len(var.activities),))
        atow = np.zeros((len(var.activities),))

        for j in range(1, len(x)):
            if x[j] == 1:
                atod = np.add(atod, time_app_data['atod'][j-1][hour])
                adow = np.add(adow, time_app_data['adow'][j-1][day])
                atow = np.add(atow, time_app_data['atow'][j-1][timeweek])

        # debug(atod)
        # debug(adow)
        # debug(atow)

        answer_atod = get_answer(atod)
        answer_adow = get_answer(adow)
        answer_atow = get_answer(atow)

        if answer_atod == y[iterator]:
            scores['atod'] += 1
        if answer_adow == y[iterator]:
            scores['adow'] += 1
        if answer_atow == y[iterator]:
            scores['atow'] += 1

        iterator += 1

    ## Normalize score
    for name in time_app_data:
        scores[name] /= len(y)

    # debug(scores)

    return scores

def extract_time_info(users_data, user_ids, mode, app_names, categories, agg=True, app_cat=None, cached=False, time_info=False, app_type='all', k_fold=K_FOLD):
    nameset = ['tod', 'dow', 'tow', 'atod', 'adow', 'atow']
    ctr_uid = 0
    dataset = {}
    debug(mode)
    if mode.lower() == 'full' or mode.lower() == 'part':
        columns = app_names
    elif mode.lower() == 'cat' or mode.lower() == 'hybrid':
        columns = categories
    report_filename = 'app_time_matrix_{}_{}_{}.csv'.format(mode, time_info, app_type)
    remove_file_if_exists(cd.soft_report + report_filename)
    debug(cd.soft_report + report_filename)
    if WRITE_HEADER:
        text = 'mode,time_info,app_type,'
        text += ','.join(nameset)
        write_to_file(cd.soft_report + report_filename, text)
    if agg:
        groups  = []
        xdata = []
        for uid, data in users_data.items():
            ctr_uid += 1
            if uid not in user_ids:
                continue
            for x in data:
                xdata.append(x)
                groups.append(ctr_uid)
        dataset['all'] = xdata
    else:
        group = None
        for uid, data in users_data.items():
            ctr_uid += 1
            if uid not in user_ids:
                continue
            dataset[uid] = data
    for uid, raw in dataset.items():
        if agg:
            username = 'all'
        else:
            username = uid
        data = np.array(raw)
        ncol = data.shape[1]
        base_col = 3
        indices = []
        indices.append(1)
        for i in range(base_col, ncol):
            indices.append(i)
        X = data[:,indices]
        y = data[:,2]
        cv, n_split = get_cv(k_fold, groups, X, y)
        i = 0
        agg_scores = {}
        for (train, test) in cv:
            i += 1
            time_app_data = build_time_matrix(username, X[train], y[train], columns, i, n_split, cached=cached)
            # debug(time_app_data['dow'])
            # for x in range(len(time_app_data['adow'])):
            #     debug(time_app_data['adow'][x], clean=True)
            scores = testing_time_app(X[test], y[test], time_app_data)
            for name, score in scores.items():
                found = agg_scores.get(name)
                if found is None:
                    found = 0
                agg_scores[name] = found + score
            break
        for name, score in agg_scores.items():
            agg_scores[name] = score / i
        text = '{},{},{},'.format(mode, time_info, app_type)
        text += ','.join(str(agg_scores[x]) for x in nameset)
        write_to_file(cd.soft_report + report_filename, text)

def extract_time_data(user_ids, mode, app_cat=None, cached=False):
    ctr_uid = 0
    global_timeline = {}
    personal_timeline = {}
    global_filename = 'global_app_{}.bin'.format(mode)
    remove_digits = str.maketrans('', '', digits)
    if not cached:
        for uid in user_ids:
            ctr_uid += 1
            with open(SOFT_FORMAT.format(cd.dataset_folder, uid)) as f:
                debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(user_ids)))
                for line in f:
                    split = line.lower().strip().split(',')
                    act = split[1]
                    app = split[2]
                    time = int(split[3])    # in ms
                    act_int = activity_to_int(act, var.activities)
                    if act_int < 0:
                        continue
                    if mode.lower() == 'full':
                        data = [app]
                    elif mode.lower() == 'part':
                        data = app.translate(remove_digits).replace(':','.').split('.')
                    elif mode.lower() == 'cat':
                        cat = app_cat.get(app.strip())
                        if cat is not None:
                            data = [cat]
                    ### Global app records
                    for app in data:
                        found = global_timeline.get(app)
                        if found is None:
                            found_time = []
                            found_act = []
                            found = (found_time, found_act)
                        (found_time, found_act) = found
                        found_time.append(time)
                        found_act.append((act_int, time))
                        global_timeline[app] = (found_time, found_act)
                        ### Personal app records
                        found = personal_timeline.get(uid)
                        if found is None:
                            found = {}
                        personal_timeline[uid] = found
                        found2 = found.get(app)
                        if found2 is None:
                            found_time = []
                            found_act = []
                            found2 = (found_time, found_act)
                        (found_time, found_act) = found2
                        found_time.append(time)
                        found_act.append((act_int, time))
                        found[app] = (found_time, found_act)
                debug(psutil.virtual_memory(), out_file=True)
        debug('Writing to file : {}'.format(global_filename))
        with open(cd.software_folder + global_filename, 'wb') as f:
            pickle.dump(global_timeline, f)
        for uid, data in personal_timeline.items():
            personal_filename = 'personal_app_{}_{}.bin'.format(mode, uid)
            debug('Writing to file : {}'.format(personal_filename))
            with open(cd.software_folder + personal_filename, 'wb') as f:
                pickle.dump(data, f)
    else:
        try:
            debug(psutil.virtual_memory(), out_file=True)
            debug("Reading global timeline cache")
            with open(cd.software_folder + global_filename, 'rb') as f:
                global_timeline = pickle.load(f)
            debug(psutil.virtual_memory(), out_file=True)
            debug("Reading personal timeline cache")
            for uid in user_ids:
                personal_filename = 'personal_app_{}_{}.bin'.format(mode, uid)
                with open(cd.software_folder + personal_filename, 'rb') as f:
                    personal_timeline[uid] = pickle.load(f)
            debug("Finished reading timeline caches")
            debug(psutil.virtual_memory(), out_file=True)
        except:
            extract_time_data(user_ids, mode, app_cat, cached=False)
    return global_timeline, personal_timeline

def init_timeslots(app, arr_len, timeslot):
    collection = []
    for i in range(arr_len):
        arr = []
        for j in range(len(var.activities)):
            arr.append(0)
        collection.append(arr)
    timeslot[app] = collection
    return timeslot

def time_slots_extraction(timeline, uid, cached=True):
    time_of_day     = {} ### 24 hour time slots
    day_of_week     = {} ### 7 day time slots
    time_of_week    = {} ### 24H x 7D time slots

    act_time_of_day     = {} ### 24 hour time slots
    act_day_of_week     = {} ### 7 day time slots
    act_time_of_week    = {} ### 24H x 7D time slots

    tod_file = TIME_CACHE.format('tod', uid)
    dow_file = TIME_CACHE.format('dow', uid)
    tow_file = TIME_CACHE.format('tow', uid)

    atod_file = TIME_CACHE.format('atod', uid)
    adow_file = TIME_CACHE.format('adow', uid)
    atow_file = TIME_CACHE.format('atow', uid)

    if cached:
        with open(cd.soft_users_cache + tod_file, 'rb') as f:
            time_of_day = pickle.load(f)
        if len(time_of_day) == 0:
            time_slots_extraction(timeline, uid, cached=False)

        with open(cd.soft_users_cache + dow_file, 'rb') as f:
            day_of_week = pickle.load(f)
        if len(day_of_week) == 0:
            time_slots_extraction(timeline, uid, cached=False)

        with open(cd.soft_users_cache + tow_file, 'rb') as f:
            time_of_week = pickle.load(f)
        if len(time_of_week) == 0:
            time_slots_extraction(timeline, uid, cached=False)

        with open(cd.soft_users_cache + atod_file, 'rb') as f:
            act_time_of_day = pickle.load(f)
        if len(act_time_of_day) == 0:
            time_slots_extraction(timeline, uid, cached=False)

        with open(cd.soft_users_cache + adow_file, 'rb') as f:
            act_day_of_week = pickle.load(f)
        if len(act_day_of_week) == 0:
            time_slots_extraction(timeline, uid, cached=False)

        with open(cd.soft_users_cache + atow_file, 'rb') as f:
            act_time_of_week = pickle.load(f)
        if len(act_time_of_week) == 0:
            time_slots_extraction(timeline, uid, cached=False)
    else:
        debug('Extract time slots: {}'.format(uid), out_file=True)
        debug(psutil.virtual_memory(), out_file=True)

        for app, found in timeline.items():
            (found_time, found_act) = found
            ### Init data time
            tod = []
            for i in range(24):
                tod.append(0)
            dow = []
            for i in range(7):
                dow.append(0)
            tow = []
            for i in range(7*24):
                tow.append(0)
            time_of_day[app] = tod
            day_of_week[app] = dow
            time_of_week[app] = tow
            ### Init data time act
            act_time_of_day = init_timeslots(app, 24, act_time_of_day)
            act_day_of_week = init_timeslots(app, 7, act_day_of_week)
            act_time_of_week = init_timeslots(app, 24*7, act_time_of_week)
            ### Sort the time
            time_data = sorted(time_data)
            # duration = time_data[len(time_data)-1] - time_data[0]
            ### Threshold is N hour
            # if float(duration) / var.MILI / var.HOUR > 1.0:
            #     debug('{} : {}'.format(app, duration))
            ### App and Time
            for time in time_data:
                date = datetime.fromtimestamp(time / 1e3)
                day = date.weekday()
                hour = date.hour
                timeweek = day*24 + hour
                time_of_day[app][hour] += 1
                day_of_week[app][day] += 1
                time_of_week[app][timeweek] += 1
            ### App, Act, and Time
            for (act, time) in act_data:
                if act == -1:
                    continue
                date = datetime.fromtimestamp(time / 1e3)
                day = date.weekday()
                hour = date.hour
                timeweek = day*24 + hour
                act_time_of_day[app][hour][act] += 1
                act_day_of_week[app][day][act] += 1
                act_time_of_week[app][timeweek][act] += 1
        # Dump into file
        with open(cd.soft_users_cache + tod_file, 'wb') as f:
            pickle.dump(time_of_day, f)
        with open(cd.soft_users_cache + dow_file, 'wb') as f:
            pickle.dump(day_of_week, f)
        with open(cd.soft_users_cache + tow_file, 'wb') as f:
            pickle.dump(time_of_week, f)
        with open(cd.soft_users_cache + atod_file, 'wb') as f:
            pickle.dump(act_time_of_day, f)
        with open(cd.soft_users_cache + adow_file, 'wb') as f:
            pickle.dump(act_day_of_week, f)
        with open(cd.soft_users_cache + atow_file, 'wb') as f:
            pickle.dump(act_time_of_week, f)
    debug('Finished time slot extractions', out_file=True)
    debug(psutil.virtual_memory(), out_file=True)
    return time_of_day, day_of_week, time_of_week, act_time_of_day, act_day_of_week, act_time_of_week

def timeline_report(mode, uid, categories=None, time_of_day=None, day_of_week=None, time_of_week=None):
    texts = []
    if time_of_day is not None:
        for app, arr in time_of_day.items():
            if mode.lower() == 'cat':
                text = '{},{}'.format(categories[app], (','.join(str(x) for x in arr)))
            else:
                text = '{},{}'.format(app, (','.join(str(x) for x in arr)))
            texts.append(text)
        write_to_file_buffered(REPORT_TOD_NAME.format(cd.soft_report, mode, uid), texts)
        del texts[:]
    if day_of_week is not None:
        for app, arr in day_of_week.items():
            if mode.lower() == 'cat':
                text = '{},{}'.format(categories[app], (','.join(str(x) for x in arr)))
            else:
                text = '{},{}'.format(app, (','.join(str(x) for x in arr)))
            texts.append(text)
        write_to_file_buffered(REPORT_DOW_NAME.format(cd.soft_report, mode, uid), texts)
        del texts[:]
    if time_of_week is not None:
        for app, arr in time_of_week.items():
            if mode.lower() == 'cat':
                text = '{},{}'.format(categories[app], (','.join(str(x) for x in arr)))
            else:
                text = '{},{}'.format(app, (','.join(str(x) for x in arr)))
            texts.append(text)
        write_to_file_buffered(REPORT_TOW_NAME.format(cd.soft_report, mode, uid), texts)
        del texts[:]

def extract_app_statistics(X, y, mode, uid, sort_mode, weight_mode, app_names=None, categories=None, cached=True, counter=0, length=0):
    acts_app = []   ### For every activities it would have a dictionary
    if cached:
        try:
            filename = cd.soft_statistics_folder + APP_STATS_NAME.format(mode, uid, sort_mode, weight_mode, counter, length)
            with open(filename, 'rb') as f:
                acts_app = pickle.load(f)
        except:
            extract_app_statistics(X, y, mode, uid, sort_mode, weight_mode, app_names, categories, cached=False)
    else:
        frequencies = []    ### consist of {} -- dict of (app name and frequency score)
        entropies = {}      ### dict of (app name and entropy score)
        cond_s = []         ### To extract X in each activity
        Xss = []            ### Store X in each activity

        names = None
        if categories is not None:
            names = categories
        else:
            names = app_names

        for i in range(len(var.activities)):
            cond_s.append(y == i)       ### Compare the activity label with current activity
            Xss.append(X[cond_s[i]])

        ### Extract basic stats for each app
        fs = []
        for i in range(len(var.activities)):
            f = np.sum(Xss[i], axis=0)
            fs.append(f)
        ### Extract frequencies
        fs = np.array(fs)
        fxs = []
        for i in range(len(var.activities)):
            fx = np.divide(fs[i], sum(fs[i]))
            if sum(fs[i]) > 0:
                fxs.append(fx)
            else:
                fxs.append(None)
            f_dict = {}
            for j in range(len(fx)):
                if fx[j] > 0:
                    # name = names[j]
                    name = j
                    f_dict[name] = fx[j]
            frequencies.append(f_dict)
        ### Extract entropies
        total = np.sum(fs, axis=0)
        fds = []
        for i in range(len(var.activities)):
            fd = np.divide(fs[i], total)
            fds.append(fd)
        fds = np.array(fds)
        fds = np.transpose(fds)
        for i in range(len(fds)):
            e = entropy(fds[i], len(fds[i]))
            # name = names[i]
            name = i
            entropies[name] = e
        ### Build the "acts_app"
        for i in range(len(var.activities)):
            act_dict = {}
            acts_app.append(act_dict)
            for j in range(len(names)):
                # name = names[j]
                name = j
                f = frequencies[i].get(name)
                e = entropies.get(name)
                if f is None:
                    continue
                act_dict[name] = {'f': f, 'e': e}

        ### Write acts_app data into file
        filename = cd.soft_statistics_folder + APP_STATS_NAME.format(mode, uid, sort_mode, weight_mode, counter, length)
        with open(filename, 'wb') as f:
            pickle.dump(acts_app, f)
    return acts_app

def select_top_k_apps(top_k, acts_app, sorting, sort_mode=DEFAULT_SORTING):
    top_k_apps = {}
    for i in range(len(var.activities)):
        k = 0
        # debug(acts_app[i])
        # debug(sorted(acts_app[i].items(), key=lambda value: value[1]['f'], reverse=True))
        (sort, desc) = sorting[sort_mode]
        ### Sorting based on sorting schemes
        top = {}
        for app_id, value in sorted(acts_app[i].items(), key=sort, reverse=desc):
            if k >= top_k:
                break
            top[app_id] = (value, k)
            k += 1
        top_k_apps[i] = top
    return top_k_apps

def evaluate_topk_apps_various(users_data, user_ids, mode, TOPK, sorting, SORT_MODES, WEIGHT_MODES, app_names=None, categories=None, cached=True, single=True, time_window=DEFAULT_TIME_WINDOW, time_info=False, app_type='all'):
    if single:
        agg_type = 'single'
    else:
        agg_type = 'agg'
    if time_info:
        time_name = '_Time_'
    else:
        time_name = ''
    filename = cd.soft_report + REPORT_TOPK_NAME.format(agg_type, mode, time_window, time_name, date.today())
    remove_file_if_exists(filename)
    text = 'UID,Mode,Topk,Sort,Weight,TimeWindow,Acc,Train,Test,TimeInfo,AppType'
    if WRITE_HEADER:
        write_to_file(filename, text)
    for topk in TOP_K:
        for weight in WEIGHTS:
            counter = 0
            for sort in SORTS:
                if weight == 'g' and counter > 0:
                    ### No need to repeat different sorting for general
                    debug('No repeat for general weighting')
                    continue
                debug('topk: {}, weight: {}, sort: {}'.format(topk, weight, sort), out_file=True)
                debug(psutil.virtual_memory(), out_file=True)
                evaluate_topk_apps(users_data, user_ids, mode, topk, sorting, sort_mode=sort, weight_mode=weight, app_names=app_names, categories=categories, cached=cached, single=single, time_window=time_window, app_type=app_type)
                counter += 1

def evaluate_topk_apps(users_data, user_ids, mode, topk, sorting, sort_mode=DEFAULT_SORTING, weight_mode=DEFAULT_WEIGHTING, app_names=None, categories=None, cached=True, single=True, time_window=DEFAULT_TIME_WINDOW, time_info=False, app_type='all'):
    ctr_uid = 0
    texts = []
    if time_info:
        time_name = '_Time_'
    else:
        time_name = ''
    if single:
        agg_type = 'single'
        for uid, data in users_data.items():
            ctr_uid += 1
            debug('SORT: {}, WEIGHT: {}, Top-K: {}'.format(sort_mode, weight_mode, topk))
            debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(users_data)), out_file=True)
            debug('#Rows: {}'.format(len(data)), out_file=True)
            if uid not in user_ids:
                continue
            output = soft_evaluation(data, uid, mode, topk, sorting, sort_mode=sort_mode, weight_mode=weight_mode, app_names=app_names, categories=categories, cached=cached)
            texts.append('{},{},{},{},{},{},{},{},{},{},{}'.format(uid, mode, topk, sort_mode, weight_mode, time_window, output['acc'], output['time_train'], output['time_test'], time_info, app_type))
    else:
        dataset = []
        groups  = []
        agg_type = 'agg'
        for uid, data in users_data.items():
            ctr_uid += 1
            if uid not in user_ids:
                continue
            for x in data:
                dataset.append(x)
                groups.append(ctr_uid)
        uid = 'ALL'
        output = soft_evaluation(dataset, uid, mode, topk, sorting, sort_mode=sort_mode, weight_mode=weight_mode, app_names=app_names, categories=categories, cached=cached, groups=groups)
        texts.append('{},{},{},{},{},{},{},{},{},{},{}'.format(uid, mode, topk, sort_mode, weight_mode, time_window, output['acc'], output['time_train'], output['time_test'], time_info, app_type))
    filename = cd.soft_report + REPORT_TOPK_NAME.format(agg_type, mode, time_window, time_name, date.today())
    write_to_file_buffered(filename, texts)

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    debug('--- Program Started ---', out_file=True)
    Time_Info = [
        False
        # True, False
    ]
    PCA = [
        False
        # False, True
    ]
    MODE = [
        'Full', 'Cat'
    ]   ## 'Full', 'Part', 'Cat', 'Hybrid'
    ## 'Full', 'Cat'

    APP_TYPE = [
        'fore', 'all'
    ]   ## 'fore', 'back', 'all'

    TOP_K = [
        # 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
        # 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
        10, 20, 30, 40, 50, 60
    ]   ## 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100

    SORTS = [
        'f', 'ef'
    ]   ## 'f', 'ef', 'erf'

    WEIGHTS = [
        'g', 'w', 'we', 'wef'
    ]   ## 'g', 'w', 'we', 'wef', 'werf'

    TIME_WINDOWS = [
        1000
    ]   ## 0, 1, 100, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000

    ### Init sorting mechanism
    sorting = init_sorting_schemes()
    ### Init user ids
    user_ids  = init()

    ### Stop words for specific app names
    stop_words = init_stop_words(STOP_FILENAME)

    for mode in MODE:
        debug('Mode is : {}'.format(mode))
        for app_type in APP_TYPE:
            ### Apps categories
            categories = None
            app_cat = None
            if mode.lower() == 'cat' or mode.lower() == 'hybrid':
                categories, app_cat = init_app_category(mode, app_type)
                debug('len(categories): {}'.format(len(categories)))
                debug('len(app_cat): {}'.format(len(app_cat)))

            ### Transform original input into training and testing dataset
            ## ToDo add hybrid in app_names extraction -- and then transform dataset also add hybrid
            app_names = get_all_apps(user_ids, stop_words, mode, app_type, write=True, cached=True)
            debug('len(app_names): {}'.format(len(app_names)))

            for time in TIME_WINDOWS:
                debug('Time window: {}'.format(time))
                debug(psutil.virtual_memory(), out_file=True)
                users_data = {}
                for time_info in Time_Info:
                    debug('time_info is : {}'.format(time_info))
                    ### Read dataset for the experiments
                    users_data = transform_dataset(user_ids, app_names, mode, write=True, categories=categories, app_cat=app_cat, cached=True, time_window=time, time_info=time_info, app_type=app_type)
                    debug('Finished transforming all data: {} users'.format(len(users_data)), out_file=True)
                    debug(psutil.virtual_memory(), out_file=True);

                    for pca in PCA:
                        debug('pca is : {}'.format(pca))
                        ## Generate testing report using machine learning evaluation
                        # generate_testing_report(users_data, user_ids, mode, clear_data=False, categories=categories, cached=True, agg=True, time_window=time, pca=pca, time_info=time_info, app_type=app_type)
                        # debug(psutil.virtual_memory(), out_file=True)

                ### Top-k apps evaluation
                # evaluate_topk_apps_various(users_data, user_ids, mode, TOP_K, sorting, SORTS, WEIGHTS, app_names=app_names, categories=categories, cached=False, single=False, time_window=time, app_type=app_type)

                ### Apps X Time evaluation
                extract_time_info(users_data, user_ids, mode, app_names, categories, agg=True, app_cat=app_cat, cached=True, time_info=time_info, app_type=app_type, k_fold=K_FOLD)

                ### Clean memory
                debug('Clearing memory', out_file=True)
                users_data.clear()
                gc.collect()
                debug(psutil.virtual_memory(), out_file=True)
            ### Extract time of each apps
            # global_timeline = None
            # personal_timeline = None
            # global_timeline, personal_timeline = extract_time_data(user_ids, mode, app_cat=app_cat, cached=False)
            # ### Global timeline
            # time_of_day, day_of_week, time_of_week, act_time_of_day, act_day_of_week, act_time_of_week = time_slots_extraction(global_timeline, 'ALL', cached=False)
            # debug(act_day_of_week)
            # timeline_report(mode, 'GLOBAL', categories=categories, time_of_day=time_of_day, day_of_week=day_of_week, time_of_week=time_of_week)
            # ### Personal timeline
            # for uid in user_ids:
            #     timeline = None
            #     if personal_timeline is not None:
            #         timeline = personal_timeline[uid]
                # time_of_day, day_of_week, time_of_week, act_time_of_day, act_day_of_week, act_time_of_week = time_slots_extraction(timeline, uid, cached=False)
                # timeline_report(mode, uid, categories=categories, time_of_day=time_of_day, day_of_week=day_of_week, time_of_week=time_of_week)

            ### Extract time info from cached data

    ### Create a tuple for software view model by transforming raw data
    ### {frequency, entropy, entropy_frequency}
    debug('--- Program Finished ---', out_file=True)