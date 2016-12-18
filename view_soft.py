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

### Supporting files
CATEGORY_NAME           = 'category_lookup.csv'
APP_CATEGORY            = 'app_category.csv'
STOP_FILENAME           = 'stop_app.txt'
APP_NAME_LIST           = 'app_{}.csv'

### Intermediate binary file
USERS_DATA_NAME         = 'users_data_{}_{}_{}.bin'                         ## [Mode] [UID] [Time Window]
APP_STATS_NAME          = 'acts_stats_{}_{}_SORT[{}]_WEIGHT[{}]_{}_{}.bin'  ## [Mode] [UID] [Sorting Mode] [Weighting Mode] [Counter] [Length]

### Software dataset and intermediate
SOFT_FORMAT             = '{}/{}_soft.csv'                                  ## Original app data    [Directory] [UID]
SOFT_PROCESSED          = '{}/{}_soft_{}_{}.csv'                            ## Processed: part name [Directory] [UID] [Mode] [Time Window]

### Reports
REPORT_NAME             = '{}/soft_report_{}_{}_{}_{}.csv'                  ## [Directory] [agg/single] [Mode] [Time Window] [Today]
REPORT_TOPK_NAME        = 'soft_report_topk_{}_{}_{}_{}.csv'                ## [single/agg] [full/part/cat] [Time Window] [Today]

REPORT_TOD_NAME         = '{}/soft_report_tod_{}_{}.csv'                    ## [Directory] [Mode] [UID]
REPORT_DOW_NAME         = '{}/soft_report_dow_{}_{}.csv'                    ## [Directory] [Mode] [UID]
REPORT_TOW_NAME         = '{}/soft_report_tow_{}_{}.csv'                    ## [Directory] [Mode] [UID]

APP_F_THRESHOLD         = 1000  ## Minimum occurrence of the app throughout the dataset

K_FOLD                  = 5

DEFAULT_SORTING         = 'f'       ### ef: entropy frequency, f: frequency, erf: entropy*sqrt(freqency)
DEFAULT_WEIGHTING       = 'g'       ### g: general (unweighted), w: rank on list, f: frequency, e: 1-entropy, ef: e*f, erf: e*sqrt(f)
DEFAULT_TIME_WINDOW     = 1000      ### in ms

COLUMN_NAMES            = 'UID,Classifier,Accuracy,TrainTime(s),TestTime(s),Mode,TimeWindow'

"""
@Initialization methods
"""
def init_stop_words(stop_app_filename):
    stop_words = []
    with open(stop_app_filename, 'r') as fr:
        for line in fr:
            stop_words.append(line.strip())
    return stop_words

def init_app_category(mode):
    filename = cd.software_folder + APP_NAME_LIST.format(mode)
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
def transform_dataset(user_ids, app_names, mode, write=False, categories=None, app_cat=None, cached=False, time_window=0):
    ### If not cached
    remove_digits = str.maketrans('', '', digits)
    users_data = {}
    ctr_uid = 0
    for uid in user_ids:
        ctr_uid += 1
        lines = []
        filename = SOFT_FORMAT.format(cd.dataset_folder, uid)
        debug('Transforming : {} [{}/{}]'.format(filename, ctr_uid, len(user_ids)), callerid=get_function_name(), out_file=True)
        ### Load cache
        user_data = None
        binary_filename = USERS_DATA_NAME.format(mode, uid, time_window)
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
                        app_dist.append(0)
                elif mode.lower() == 'cat' or mode.lower() == 'hybrid':
                    for i in range(len(categories)):
                        app_dist.append(0)
                for line in fr:
                    split = line.lower().strip().split(',')
                    uid = int(split[0])
                    act = split[1]
                    app = split[2]
                    time = int(split[3])    # in ms
                    act_int = activity_to_int(act, var.activities)
                    # print(act_int)
                    date = datetime.fromtimestamp(time / 1e3)
                    if mode.lower() == 'cat' or mode.lower() == 'hybrid':
                        cat = app_cat.get(app.strip())
                        if cat is not None:
                            try:
                                app_dist[cat] = 1
                            except Exception as ex:
                                debug(cat)
                                debug(ex, get_function_name())
                    elif mode.lower() == 'full':
                        try:
                            idx = app_names.index(app)
                            if idx != -1:
                                app_dist[idx] = 1
                        except:
                            ### Because some app names are deleted to save resources
                            pass
                    elif mode.lower() == 'part':
                        app_split = app.translate(remove_digits).replace(':','.').split('.')
                        for app_id in app_split:
                            try:
                                idx = app_names.index(app_id)
                                if idx != -1:
                                    app_dist[idx] = 1
                            except:
                                ### Because some app names are deleted to save resources
                                pass
                    if abs(time - previous_time) >= time_window or ctr == num_lines-1:
                        if sum(app_dist) > 0:
                            # soft = (','.join(str(x) for x in app_dist))
                            # text = '{},{},{}'.format(uid, act_int, soft)
                            # lines.append(text)
                            ### label is put in the first column
                            data = []
                            data.append(uid)
                            data.append(time)
                            data.append(act_int)
                            data.extend(app_dist)
                            user_data.append(data)
                            # debug(data)
                            ## Reset app distributions
                            app_dist = []
                            if categories is None:
                                for i in range(len(app_names)):
                                    app_dist.append(0)
                            else:
                                for i in range(len(categories)):
                                    app_dist.append(0)
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
def get_all_apps(user_ids, stop_words, mode, write=False, cached=False):
    ### Cached
    if mode.lower() == 'full' or mode.lower() == 'part':
        app_filename = cd.software_folder + APP_NAME_LIST.format(mode.lower())
    else:
        app_filename = cd.software_folder + APP_NAME_LIST.format('full')
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
def testing(dataset, uid, mode, cached=True, groups=None, time_window=DEFAULT_TIME_WINDOW):
    debug('Processing: {}'.format(uid))
    dataset = np.array(dataset)
    clfs = classifier_list()
    try:
        # debug(dataset.shape)
        # debug(dataset[0])
        ncol = dataset.shape[1]
        X = dataset[:,3:ncol] # Remove index 0 (uid), index 1 (time), and index 2 (activities)
        y = dataset[:,2]
        texts = []
        info = {}
        info['uid'] = uid
        for name, clf in clfs.items():
            debug(name)
            info['clf_name'] = name
            output = evaluation(X, y, clf, k_fold=K_FOLD, info=info, cached=cached, mode=mode, groups=groups, time_window=time_window)
            acc = output['acc']
            time_train = output['time_train']
            time_test = output['time_test']
            text = '{},{},{},{},{},{},{}'.format(uid, name, acc, time_train, time_test, mode, time_window)
            texts.append(text)
        return texts
    except Exception as ex:
        debug('Error on testing', ex)
        return None

### Generating testing report per user
def generate_testing_report(users_data, user_ids, mode, clear_data=False, categories=None, cached=True, agg=False, time_window=DEFAULT_TIME_WINDOW):
    # ### Test
    if not agg:
        debug('Evaluating application data (single)', out_file=True)
    else:
        debug('Evaluating application data (agg)', out_file=True)
    dataset = []
    groups  = []
    output = []
    output.append(COLUMN_NAMES)
    ctr_uid = 0
    for uid, data in users_data.items():
        ctr_uid += 1
        debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(users_data)), out_file=True)
        debug('#Rows: {}'.format(len(data)), out_file=True)
        if uid not in user_ids:
            continue
        if not agg:
            result = testing(data, uid, mode=mode, cached=cached, time_window=time_window)
            if result is not None:
                output.extend(result)
        else:
            for x in data:
                dataset.append(x)
                groups.append(ctr_uid)
    if agg:
        uid = 'ALL'
        result = testing(dataset, uid, mode=mode, cached=cached, groups=groups, time_window=time_window)
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
    filename = REPORT_NAME.format(cd.soft_report, agg_name, mode, time_window, date.today())
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, output)
    # debug(output)

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
                            found = []
                        found.append(time)
                        global_timeline[app] = found
                        ### Personal app records
                        found = personal_timeline.get(uid)
                        if found is None:
                            found = {}
                        personal_timeline[uid] = found
                        found2 = found.get(app)
                        if found2 is None:
                            found2 = []
                        found2.append(time)
                        found[app] = found2
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
            with open(cd.software_folder + global_filename, 'rb') as f:
                global_timeline = pickle.load(f)
            for uid in user_ids:
                personal_filename = 'personal_app_{}_{}.bin'.format(mode, uid)
                with open(cd.software_folder + personal_filename, 'rb') as f:
                    personal_timeline[uid] = pickle.load(f)
        except:
            extract_time_data(user_ids, mode, app_cat, cached=False)
    return global_timeline, personal_timeline

def time_slots_extraction(timeline):
    time_of_day     = {} ### 24 hour time slots
    day_of_week     = {} ### 7 day time slots
    time_of_week    = {} ### 24H x 7D time slots
    for app, data in timeline.items():
        ### Init data
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
        ###
        data = sorted(data)
        duration = data[len(data)-1] - data[0]
        ### Threshold is N hour
        if float(duration) / var.MILI / var.HOUR > 1.0:
            debug('{} : {}'.format(app, duration))
        for time in data:
            date = datetime.fromtimestamp(time / 1e3)
            day = date.weekday()
            hour = date.hour
            timeweek = day*24 + hour
            time_of_day[app][hour] += 1
            day_of_week[app][day] += 1
            time_of_week[app][timeweek] += 1
    return time_of_day, day_of_week, time_of_week

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
                    name = names[j]
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
            name = names[i]
            entropies[name] = e
        ### Build the "acts_app"
        for i in range(len(var.activities)):
            act_dict = {}
            acts_app.append(act_dict)
            for j in range(len(names)):
                name = names[j]
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

def evaluate_topk_apps_various(users_data, user_ids, mode, TOPK, sorting, SORT_MODES, WEIGHT_MODES, app_names=None, categories=None, cached=True, single=True, time_window=DEFAULT_TIME_WINDOW):
    if single:
        agg_type = 'single'
    else:
        agg_type = 'agg'
    filename = cd.soft_report + REPORT_TOPK_NAME.format(agg_type, mode, time_window, date.today())
    remove_file_if_exists(filename)
    text = 'UID,Mode,Topk,Sort,Weight,TimeWindow,Acc,Train,Test'
    write_to_file(filename, text)
    for topk in TOP_K:
        for sort in SORTS:
            for weight in WEIGHTS:
                evaluate_topk_apps(users_data, user_ids, mode, topk, sorting, sort_mode=sort, weight_mode=weight, app_names=app_names, categories=categories, cached=cached, single=single, time_window=time_window)

def evaluate_topk_apps(users_data, user_ids, mode, topk, sorting, sort_mode=DEFAULT_SORTING, weight_mode=DEFAULT_WEIGHTING, app_names=None, categories=None, cached=True, single=True, time_window=DEFAULT_TIME_WINDOW):
    ctr_uid = 0
    texts = []
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
            texts.append('{},{},{},{},{},{},{},{},{}'.format(uid, mode, topk, sort_mode, weight_mode, time_window, output['acc'], output['time_train'], output['time_test']))
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
        texts.append('{},{},{},{},{},{},{},{},{}'.format(uid, mode, topk, sort_mode, weight_mode, time_window, output['acc'], output['time_train'], output['time_test']))
    filename = cd.soft_report + REPORT_TOPK_NAME.format(agg_type, mode, time_window, date.today())
    write_to_file_buffered(filename, texts)

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    debug('--- Program Started ---', out_file=True)
    MODE = [
        'Full', 'Part', 'Cat', 'Hybrid'
    ]   ## 'Full', 'Part', 'Cat', 'Hybrid'

    TOP_K = [
        10
    ]   ## 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60

    SORTS = [
        'ef'
    ]   ## 'f', 'ef', 'erf'

    WEIGHTS = [
        'wef'
    ]   ## 'g', 'w', 'f', 'e', 'ef', 'erf', 'we', 'wef', 'werf'

    TIME_WINDOWS = [
        500, 1000
    ]   ## 0, 1, 100, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000

    ### Init sorting mechanism
    sorting = init_sorting_schemes()
    ### Init user ids
    user_ids  = init()

    ### Stop words for specific app names
    stop_words = init_stop_words(STOP_FILENAME)

    for mode in MODE:
        debug('Mode is : {}'.format(mode))

        ### Apps categories
        categories = None
        app_cat = None
        if mode.lower() == 'cat' or mode.lower() == 'hybrid':
            categories, app_cat = init_app_category(mode)
            debug('len(categories): {}'.format(len(categories)))
            debug('len(app_cat): {}'.format(len(app_cat)))

        ### Transform original input into training and testing dataset
        ## ToDo add hybrid in app_names extraction -- and then transform dataset also add hybrid
        app_names = get_all_apps(user_ids, stop_words, mode, write=True, cached=True)
        debug('len(app_names): {}'.format(len(app_names)))

        for time in TIME_WINDOWS:
            debug('Time window: {}'.format(time))
            ### Read dataset for the experiments
            users_data = transform_dataset(user_ids, app_names, mode, write=True, categories=categories, app_cat=app_cat, cached=True, time_window=time)
            debug('Finished transforming all data: {} users'.format(len(users_data)), out_file=True)

            ### Generate testing report using machine learning evaluation
            generate_testing_report(users_data, user_ids, mode, clear_data=False, categories=categories, cached=True, agg=True, time_window=time)

            ### Top-k apps evaluation
            # evaluate_topk_apps_various(users_data, user_ids, mode, TOP_K, sorting, SORTS, WEIGHTS, app_names=app_names, categories=categories, cached=False, single=True, time_window=time)

            ### Clean memory
            users_data.clear()
        ### Extract time of each apps
        # global_timeline, personal_timeline = extract_time_data(user_ids, mode, app_cat=app_cat, cached=True)
        # ### Global timeline
        # time_of_day, day_of_week, time_of_week = time_slots_extraction(global_timeline)
        # timeline_report(mode, 'GLOBAL', categories=categories, time_of_day=time_of_day, day_of_week=day_of_week, time_of_week=time_of_week)
        # ### Personal timeline
        # for uid in user_ids:
        #     time_of_day, day_of_week, time_of_week = time_slots_extraction(personal_timeline[uid])
        #     timeline_report(mode, uid, categories=categories, time_of_day=time_of_day, day_of_week=day_of_week, time_of_week=time_of_week)

        ### Extract statistics
        # ctr_uid = 0
        # for uid, data in users_data.items():
        #     ctr_uid += 1
        #     debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(users_data)), out_file=True)
        #     debug('#Rows: {}'.format(len(data)), out_file=True)
        #     if uid not in user_ids:
        #         continue
        #     acts_app = extract_app_statistics(data, mode, uid, app_names=app_names, categories=categories, cached=True)
        #     debug(acts_app[2])
        #     top_k_apps = select_top_k_apps(TOP_K, acts_app, sorting, SORT_MODE)
        #     # soft_evaluation(X, y, top_k_apps, WEIGHTING)
        #     for i in range(len(var.activities)):
        #         debug(var.activities[i], clean=True)
        #         top = top_k_apps[i]
        #         for app_id, value in top:
        #             debug('{},{},{}'.format(app_id, value['e'], value['f']), clean=True)
        #         print()

    ### Create a tuple for software view model by transforming raw data
    ### {frequency, entropy, entropy_frequency}
    debug('--- Program Finished ---', out_file=True)