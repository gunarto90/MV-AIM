"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.11
2016/12/01 10:22PM
"""
## ToDo !!: From the users_data --> generate activity X apps vector which shows the score of each apps regarding its {frequency, entropy, entropy-frequency, other}
"""

- Fixed bug on cached is True, when the file is not available
- Added uid parameter in apps stats and acts stats
- Added testing time in evaluation
- Added new sorting method (erf) : entropy and sqrt of frequency
- Added uid in the dataset
- Added loop in the main function for various mode
"""
import getopt
import sys
import re
import os
import json
import pickle
import operator

import numpy as np
import config_directory as cd
import config_variable as var

from datetime import datetime, date
from string import digits
from collections import defaultdict
from math import sqrt

from general import *
from evaluation import *

APP_PART_NAMES          = 'app_names_part.csv'
APP_FULL_NAMES          = 'app_names_full.csv'
CATEGORY_NAME           = 'category_lookup.csv'
APP_CATEGORY            = 'app_category.csv'
STOP_FILENAME           = 'stop_app.txt'

APP_STATS_PART          = 'app_stats_part_{}.bin'
APP_STATS_FULL          = 'app_stats_full_{}.bin'
APP_STATS_CAT           = 'app_stats_cat_{}.bin'

ACTS_STATS_PART         = 'acts_stats_part_{}.bin'
ACTS_STATS_FULL         = 'acts_stats_full_{}.bin'
ACTS_STATS_CAT          = 'acts_stats_cat_{}.bin'

SOFT_FORMAT             = '{}/{}_soft.csv'          ## Original app data
SOFT_PART_FORMAT        = '{}/{}_soft_part.csv'     ## Processed: part name
SOFT_FULL_FORMAT        = '{}/{}_soft_full.csv'     ## Processed: full name
SOFT_CATEGORY_FORMAT    = '{}/{}_soft_cat.csv'      ## Processed: category

USERS_DATA_PART_NAME    = 'users_part_data.bin'
USERS_DATA_FULL_NAME    = 'users_full_data.bin'
USERS_DATA_CAT_NAME     = 'users_cat_data.bin'

REPORT_PART_NAME        = '{}/soft_report_part_{}.csv'
REPORT_FULL_NAME        = '{}/soft_report_full_{}.csv'
REPORT_CAT_NAME         = '{}/soft_report_cat_{}.csv'

APP_F_THRESHOLD         = 1000  ## Minimum occurrence of the app throughout the dataset
TIME_WINDOW             = 1000  ## in ms

DEFAULT_SORTING         = 'f'       ### ef: entropy frequency, f: frequency, e: entropy
WEIGHTING               = 'g'       ### ef: entropy frequency, f: frequency, e: entropy, g: general (unweighted), r: rank on list

COLUMN_NAMES            = 'UID,Classifier,Accuracy,TrainTime(s),TestTime(s)'

def aggregate_apps(user_ids, full=False, categories=None, app_cat=None):
    apps_agg = []
    apps_single = {}
    get = apps_single.get
    remove_digits = str.maketrans('', '', digits)
    
    ctr_uid = 0
    for uid in user_ids:
        ctr_uid += 1
        filename = SOFT_FORMAT.format(cd.dataset_folder, uid)
        with open(filename) as fr:
            debug('Statistics : {} [{}/{}]'.format(filename, ctr_uid, len(user_ids)), callerid=get_function_name(), out_file=True)
            for line in fr:
                split = line.lower().strip().split(',')
                uid = int(split[0])
                act = split[1]
                app = split[2]
                time = int(split[3])    # in ms
                act_int = activity_to_int(act, var.activities)
                date = datetime.fromtimestamp(time / 1e3)
                if categories is not None:
                    cat = app_cat.get(app.strip())
                    if cat is not None:
                        cat_name = categories[cat]
                        apps_agg.append(cat_name)
                        found = get(cat_name)
                        if found is None:
                            found = []
                        found.append(act_int)
                        apps_single[cat_name] = found
                elif full:
                    apps_agg.append(app)
                    found = get(app)
                    if found is None:
                        found = []
                    found.append(act_int)
                    apps_single[app] = found
                else:
                    app_split = app.translate(remove_digits).replace(':','.').split('.')
                    for app_id in app_split:
                        apps_agg.append(app_id)
                        # apps_single[app_id] = get(app_id, []).append(act_int)
                        found = get(app_id)
                        if found is None:
                            found = []
                        found.append(act_int)
                        apps_single[app_id] = found
    return apps_agg, apps_single

"""
Normalizing frequency and adding entropy information
"""
def normalize_app_statistics(app_stats, acts_app):
    total = []
    for i in range(len(var.activities)):
        total.append(0)
    for i in range(len(var.activities)):
        acts_map = acts_app[i]
        total[i] = (sum(acts_map[x]['f'] for x in acts_map))
        # total = sum(acts_map[x]['f'] for x in acts_map)
    for app_id, (f, e) in app_stats.items():
        for i in range(len(var.activities)):
            acts_map = acts_app[i]
            found = acts_map.get(app_id)
            if found is None and e <= 0.0:
                continue
            if found is None:
                found = {'f':0, 'e':0.0}
            if total[i] > 0:
                found['f'] = float(found['f'])/total[i]
            else:
                found['f'] = 0.0
            found['e'] = e
            if found['f'] > 0 and found['e'] > 0:
                acts_map[app_id] = found

"""
Extracting the statistics of each app in each activity
"""
def app_statistics(stop_words, apps_single):
    ### app_stats : entropy and frequency of each app
    ### acts_app  : frequency of app in specific activity
    app_stats = {}
    acts = []
    acts_app = []
    for i in range(len(var.activities)):
        acts.append(0)
        acts_app.append({})
    # debug(acts)
    for app_id, uacts in apps_single.items():
        if app_id in stop_words:
            continue
        # debug(app_id)
        # debug(len(uacts))
        for a in uacts:
            acts[a] += 1
            acts_map = acts_app[a]
            found = acts_map.get(app_id)
            if found is None:
                found = {'f':0, 'e':0.0}
            found['f'] += 1
            acts_map[app_id] = found
        # debug(len(acts))
        # debug(entropy(acts))
        acts_str = ''.join(','+str(x) for x in acts)
        text = '{},{},{}{}'.format(app_id, len(uacts), entropy(acts, len(var.activities)), acts_str)
        app_stats[app_id] = (len(uacts), entropy(acts, len(var.activities)))
        # debug(text, clean=True)
        ### Clean array in every loop
        del acts[:]
        for i in range(len(var.activities)):
            acts.append(0)
    normalize_app_statistics(app_stats, acts_app)
    return app_stats, acts_app

def extract_statistics(user_ids, stop_words, uid, full=False, categories=None, app_cat=None, cached=False):
    ### If cached
    if cached:
        app_stats, acts_app = extract_statistics_cached(uid, full=full, categories=categories)
        if app_stats is not None and acts_app is not None:
            return app_stats, acts_app
    ### If not cached
    apps_agg, apps_single = aggregate_apps(user_ids, full=full, categories=categories, app_cat=app_cat)

    debug('len(apps_agg): {}'.format(len(apps_agg)))
    debug('len(apps_single): {}'.format(len(apps_single)))
    debug('Activities: {}'.format(var.activities))

    app_stats, acts_app = app_statistics(stop_words, apps_single)

    ### Dump the statistics (app stats)
    if categories is not None:
        filename = APP_STATS_CAT.format(uid)
    elif full:
        filename = APP_STATS_FULL.format(uid)
    else:
        filename = APP_STATS_PART.format(uid)
    with open(cd.software_folder + filename, 'wb') as f:
        pickle.dump(app_stats, f)

    ### Dump the statistics (acts stats)
    if categories is not None:
        filename = ACTS_STATS_CAT.format(uid)
    elif full:
        filename = ACTS_STATS_FULL.format(uid)
    else:
        filename = ACTS_STATS_PART.format(uid)
    with open(cd.software_folder + filename, 'wb') as f:
        pickle.dump(acts_app, f)

    return app_stats, acts_app

def extract_statistics_cached(uid, full=False, categories=None):
    app_stats = None
    acts_app = None
    ### Load the statistics (app stats)
    try:
        if categories is not None:
            filename = APP_STATS_CAT.format(uid)
        elif full:
            filename = APP_STATS_FULL.format(uid)
        else:
            filename = APP_STATS_PART.format(uid)
        with open(cd.software_folder + filename, 'rb') as f:
            app_stats = pickle.load(f)
    except Exception as ex:
        debug(ex, get_function_name())

    ### Load the statistics (acts stats)
    try:
        if categories is not None:
            filename = ACTS_STATS_CAT.format(uid)
        elif full:
            filename = ACTS_STATS_FULL.format(uid)
        else:
            filename = ACTS_STATS_PART.format(uid)
        with open(cd.software_folder + filename, 'rb') as f:
            acts_app = pickle.load(f)
    except Exception as ex:
        debug(ex, get_function_name())

    return app_stats, acts_app

def select_top_k_apps(top_k, acts_app, sorting, sort_mode=DEFAULT_SORTING):
    top_k_apps = {}        
    for i in range(len(var.activities)):
        k = 0
        # debug(acts_app[i])
        # debug(sorted(acts_app[i].items(), key=lambda value: value[1]['f'], reverse=True))
        (sort, desc) = sorting[sort_mode]
        ### Sorting based on sorting schemes
        top = []
        for app_id, value in sorted(acts_app[i].items(), key=sort, reverse=desc):
            if k >= top_k:
                break
            top.append((app_id, value))
            k += 1
        top_k_apps[i] = top
    return top_k_apps

def init_stop_words(stop_app_filename):
    stop_words = []
    with open(stop_app_filename, 'r') as fr:
        for line in fr:
            stop_words.append(line.strip())
    return stop_words

def init_categories():
    filename = cd.software_folder + CATEGORY_NAME
    categories = {}
    with open(filename) as f:
        for line in f:
            split = line.strip().split(',')
            cat_name = split[0]
            cat_id = int(split[1])
            categories[cat_id] = cat_name
    return categories

def init_app_category():
    filename = cd.software_folder + APP_CATEGORY
    app_cat = {}
    with open(filename) as f:
        for line in f:
            split = line.strip().split(',')
            app_name = split[0]
            cat_id = int(split[1])
            cat_name = split[2]
            app_cat[app_name] = cat_id
    return app_cat

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
def transform_dataset(user_ids, app_names, write=False, full=False, categories=None, app_cat=None, cached=False):
    ### Cached
    if cached:
        users_data = read_dataset_from_file(full=full, categories=categories)
        if users_data is not None:
            return users_data
    ### If not cached
    remove_digits = str.maketrans('', '', digits)
    users_data = {}
    ctr_uid = 0
    if categories is not None:
        agg_filename = SOFT_CATEGORY_FORMAT.format(cd.software_folder, 'ALL')
    elif full:
        agg_filename = SOFT_FULL_FORMAT.format(cd.software_folder, 'ALL')
    else:
        agg_filename = SOFT_PART_FORMAT.format(cd.software_folder, 'ALL')
    remove_file_if_exists(agg_filename)
    for uid in user_ids:
        ctr_uid += 1
        lines = []
        user_data = []
        users_data[uid] = user_data
        filename = SOFT_FORMAT.format(cd.dataset_folder, uid)
        ctr = 0
        with open(filename) as fr:
            debug('Transforming : {} [{}/{}]'.format(filename, ctr_uid, len(user_ids)), callerid=get_function_name(), out_file=True)
            previous_time = 0
            for line in fr:
                split = line.lower().strip().split(',')
                uid = int(split[0])
                act = split[1]
                app = split[2]
                time = int(split[3])    # in ms
                act_int = activity_to_int(act, var.activities)
                # print(act_int)
                date = datetime.fromtimestamp(time / 1e3)
                if time != previous_time:
                    app_dist = []
                if categories is None:
                    for i in range(len(app_names)):
                        app_dist.append(0)
                    if not full:
                        app_split = app.translate(remove_digits).replace(':','.').split('.')
                        for app_id in app_split:
                            try:
                                idx = app_names.index(app_id)
                                if idx != -1:
                                    app_dist[idx] += 1
                            except:
                                ### Because some app names are deleted to save resources
                                pass
                    else:
                        try:
                            idx = app_names.index(app)
                            if idx != -1:
                                app_dist[idx] += 1
                        except:
                            ### Because some app names are deleted to save resources
                            pass
                else:
                    for i in range(len(categories)):
                        app_dist.append(0)
                    cat = app_cat.get(app.strip())
                    if cat is not None:
                        app_dist[cat] += 1
                if time != previous_time:
                    if sum(app_dist) > 0:
                        soft = (','.join(str(x) for x in app_dist))
                        text = '{},{},{}'.format(uid, act_int, soft)
                        lines.append(text)
                        ### label is put in the first column
                        data = []
                        data.append(uid)
                        data.append(act_int)
                        data.extend(app_dist)
                        user_data.append(data)
                    ### Finally update the previous time to match current time
                    previous_time = time
                ctr += 1
                if ctr % 100000 == 0:
                    debug('Processing {:,} lines'.format(ctr), out_file=True)
            debug('len(texts): {}'.format(len(lines)))
            debug('len(file) : {}'.format(ctr))
            if write:
                ### Write to each user's file
                if categories is not None:
                    filename = SOFT_CATEGORY_FORMAT.format(cd.software_folder, uid)
                elif full:
                    filename = SOFT_FULL_FORMAT.format(cd.software_folder, uid)
                else:
                    filename = SOFT_PART_FORMAT.format(cd.software_folder, uid)
                remove_file_if_exists(filename)
                write_to_file_buffered(filename, lines, buffer_size=1000)
                ### Write to aggregated file
                write_to_file_buffered(agg_filename, lines, buffer_size=1000)
                del lines[:]
    debug('Started writing all app and users data into binary files', out_file=True)
    if write:
        if categories is None:
            if full:
                filename = USERS_DATA_FULL_NAME
            else:
                filename = USERS_DATA_PART_NAME
        else:
            filename = USERS_DATA_CAT_NAME
        with open(cd.software_folder + filename, 'wb') as f:
            pickle.dump(users_data, f)
    debug('Finished writing all app and users data into binary files', out_file=True)
    return users_data

"""
Generating dataset from cached file (after the first time)
"""
def read_dataset_from_file(full=False, categories=None):
    debug('Started reading dataset from file')
    users_data = {}
    if categories is None:
        if full:
            filename = USERS_DATA_FULL_NAME
        else:
            filename = USERS_DATA_PART_NAME
    else:
        filename = USERS_DATA_CAT_NAME
    try:
        with open(cd.software_folder + filename, 'rb') as f:
            users_data = pickle.load(f)
    except:
        pass
    debug('Finished reading dataset from file')
    return users_data

"""
Generating all apps names from raw file for the first time
"""
def get_all_apps(user_ids, stop_words, write=False, split=True, cached=False):
    ### Cached
    if cached:
        app_names = get_all_apps_buffered(stop_words, full=not split)
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
                    if split:
                        app_split = app.translate(remove_digits).replace(':','.').split('.')
                        for app_id in app_split:
                            app_names[app_id] += 1
                    else:
                        split = app.split(':')
                        app_id = split[0]
                        app_names[app_id] += 1
        except Exception as ex:
            debug('Exception: {}'.format(ex), callerid='get_all_apps')
    for app in stop_words:
        app_names.pop(app, None)
    debug('Finished get all app names')
    if write:
        if split:
            filename = cd.software_folder + APP_PART_NAMES
        else:
            filename = cd.software_folder + APP_FULL_NAMES
        remove_file_if_exists(filename)
        texts = []
        for k, v in app_names.items():
            texts.append('{},{}'.format(k,v))
        write_to_file_buffered(filename, texts)
    return app_names.keys()

"""
Generating all apps names from cached file (after the first time)
"""
def get_all_apps_buffered(stop_words, full=False):
    if full is False:
        filename = cd.software_folder + APP_PART_NAMES
    else:
        filename = cd.software_folder + APP_FULL_NAMES
    app_names = []
    try:
        with open(filename) as fr:
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
    return app_names

### label is put in the first column
def testing(dataset, uid, cached=True, mode='Default'):
    debug('Processing: {}'.format(uid))
    dataset = np.array(dataset)
    clfs = classifier_list()
    # print(dataset.shape)
    ncol = dataset.shape[1]
    X = dataset[:,2:ncol] # Remove index 0 (uid) and index 1 (activities)
    y = dataset[:,0]
    texts = []
    info = {}
    info['uid'] = uid
    for name, clf in clfs.items():
        debug(name)
        info['clf_name'] = name
        output = evaluation(X, y, clf, info=info, cached=cached, mode=mode)
        acc = output['acc']
        time_train = output['time_train']
        time_test = output['time_test']
        text = '{},{},{},{},{}'.format(uid, name, acc, time_train, time_test)
        texts.append(text)
    return texts

### Generating testing report per user
def generate_testing_report_single(users_data, user_ids, clear_data=False, full=False, categories=None):
    # ### Test
    debug('Evaluating application data (single)', out_file=True)
    output = []
    output.append(COLUMN_NAMES)
    ctr_uid = 0
    if categories is not None:
        mode = 'cat'
    elif full:
        mode = 'full'
    else:
        mode = 'part'
    for uid, data in users_data.items():
        ctr_uid += 1
        debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(users_data)), out_file=True)
        debug('#Rows: {}'.format(len(data)), out_file=True)
        if uid not in user_ids:
            continue
        result = testing(data, uid, cached=False, mode=mode)
        output.extend(result)
    if clear_data:
        try:
            users_data.clear()
        except Exception as ex:
            debug(ex, get_function_name())
    if categories is not None:
        filename = REPORT_CAT_NAME.format(cd.report_folder, date.today())
    elif full:
        filename = REPORT_FULL_NAME.format(cd.report_folder, date.today())
    else:
        filename = REPORT_PART_NAME.format(cd.report_folder, date.today())
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, output)
    # debug(output)

### Generating testing report for all users
def generate_testing_report_agg(users_data, clear_data=False):
    debug('Evaluating application data (agg)', out_file=True)
    pass

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    debug('--- Program Started ---', out_file=True)
    MODE = [
        'Full', 'Part', 'Cat'
    ]  ## 'Full', 'Part', 'Cat'

    TOP_K = 10
    SORT_MODE = 'erf'

    ### Init sorting mechanism
    sorting = init_sorting_schemes()
    ### Init user ids
    user_ids  = init()

    ### Stop words for specific app names
    stop_words = init_stop_words(STOP_FILENAME)

    for mode in MODE:
        if mode == 'Full' or mode == 'Cat':
            full_app_name = True
        else:
            full_app_name = False

        ### Apps categories
        categories = None
        app_cat = None
        if mode == 'Cat':
            categories = init_categories()
            app_cat = init_app_category()

        ### Transform original input into training and testing dataset
        app_names = get_all_apps(user_ids, stop_words, write=True, split=not full_app_name, cached=True) ## Only for 1st time (dict)
        debug('len(app_names): {}'.format(len(app_names)))

        ### Read dataset for the experiments
        users_data = transform_dataset(user_ids, app_names, write=True, full=full_app_name, categories=categories, app_cat=app_cat, cached=False)   ## Only for 1st time (dict)
        debug('Finished transforming all data: {} users'.format(len(users_data)), out_file=True)

        ### Generate testing report using machine learning evaluation
        # generate_testing_report_single(users_data, user_ids, clear_data=False, full=full_app_name, categories=categories)
        # generate_testing_report_agg(users_data, clear_data=False)

        ### Extract statistics
        # uids = {}
        # uids['ALL'] = user_ids
        # for uid in user_ids:
        #     uids[uid] = [uid]
        # for uid, arr in uids.items():
        #     app_stats, acts_app = extract_statistics(arr, stop_words, uid, full=full_app_name, categories=categories, app_cat=app_cat, cached=True)
        #     debug(uid, clean=True)
        #     # debug(app_stats)
        #     # debug()
            # debug(acts_app)
            # top_k_apps = select_top_k_apps(TOP_K, acts_app, sorting, SORT_MODE)
            # if categories is not None:
            #     filename = SOFT_CATEGORY_FORMAT.format(cd.software_folder, uid)
            # elif full:
            #     filename = SOFT_FULL_FORMAT.format(cd.software_folder, uid)
            # else:
            #     filename = SOFT_PART_FORMAT.format(cd.software_folder, uid)
            # dataset = np.genfromtxt(filename, delimiter=',')
            # debug(dataset.shape)
            # ncol = dataset.shape[1]
            # X = dataset[:,2:ncol] # Remove index 0 (uid) and index 1 (activities)
            # y = dataset[:,0]
            # soft_evaluation(X, y, top_k_apps, WEIGHTING)
            # for i in range(len(var.activities)):
            #     debug(var.activities[i], clean=True)
            #     top = top_k_apps[i]
            #     for app_id, value in top:
            #         debug('{},{},{}'.format(app_id, value['e'], value['f']), clean=True)
            #     print()

    ### Create a tuple for software view model by transforming raw data
    ### {frequency, entropy, entropy_frequency}
    debug('--- Program Finished ---', out_file=True)

### Reference
# A = [0, 1, 2, 5]
# B = [1, 0, 1, 0]
# C = list(map(operator.add, A, B))
# debug(A)
# debug(B)
# debug(C)