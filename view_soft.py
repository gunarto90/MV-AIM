"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.8
2016/11/28 03:34PM
"""
import getopt
import sys
import re
import os
import json
import pickle

import numpy as np
import config_directory as cd
import config_variable as var

from datetime import datetime, date
from string import digits
from collections import defaultdict

from general import *
from evaluation import *

SOFT_FORMAT     = '{}/{}_soft.csv'
STOP_FILENAME   = 'stop_app.txt'
APP_NAMES       = 'app_names.csv'
ALL_APP_NAME    = 'all_app.bin'
USERS_DATA_NAME = 'users_data.bin'
APP_F_THRESHOLD = 1000
TOP_K           = 5
WEIGHTING       = 'g'   ### ef: entropy frequency, f: frequency, e: entropy, g: general (unweighted)
SORTING         = 'ef'  ### ef: entropy frequency, f: frequency, e: entropy

COLUMN_NAMES    = 'UID,Classifier,Accuracy,Time(s)'

def read_soft_file_agg(dataset_folder, user_ids):
    apps_agg = []
    apps_single = {}
    remove_digits = str.maketrans('', '', digits)
    
    for uid in user_ids:
        filename = SOFT_FORMAT.format(cd.dataset_folder, uid)
        with open(filename) as fr:
            # debug(filename, callerid=get_function_name())
            for line in fr:
                split = line.strip().split(',')
                uid = int(split[0])
                act = split[1]
                app = split[2]
                time = int(split[3])    # in ms
                act_int = activity_to_int(act, activities)
                date = datetime.fromtimestamp(time / 1e3)
                app_split = app.translate(remove_digits).replace(':','.').split('.')
                for app_id in app_split:
                    apps_agg.append(app_id)
                    get = apps_single.get
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
def normalize_app_statistics(app_stats, acts_app, activities):
    total = []
    for i in range(len(activities)):
        total.append(0)
    for i in range(len(activities)):
        acts_map = acts_app[i]
        total[i] = (sum(acts_map[x]['f'] for x in acts_map))
        # total = sum(acts_map[x]['f'] for x in acts_map)
    for app_id, (f, e) in app_stats.items():
        for i in range(len(activities)):
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
def app_statistics(stop_words, activities, apps_single):
    ### app_stats : entropy and frequency of each app
    ### acts_app  : frequency of app in specific activity
    app_stats = {}
    acts = []
    acts_app = []
    for i in range(len(activities)):
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
        text = '{},{},{}{}'.format(app_id, len(uacts), entropy(acts, len(activities)), acts_str)
        app_stats[app_id] = (len(uacts), entropy(acts, len(activities)))
        # debug(text, clean=True)
        ### Clean array in every loop
        del acts[:]
        for i in range(len(activities)):
            acts.append(0)
    normalize_app_statistics(app_stats, acts_app, activities)
    return app_stats, acts_app

def select_top_k_apps(top_k, activities, acts_app):
    top_k_apps = {}        
    for i in range(len(activities)):
        k = 0
        # debug(acts_app[i])
        # debug(sorted(acts_app[i].items(), key=lambda value: value[1]['f'], reverse=True))
        (sort, desc) = sorting[SORTING]
        ### Sorting based on sorting schemes
        top = []
        for app_id, value in sorted(acts_app[i].items(), key=sort, reverse=desc):
            if k >= TOP_K:
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

def init_sorting_schemes():
    sorting = {}
    ### Sorting mechanism, reverse (descending order) status [True/False]
    sorting['ef']   = (lambda value: (1-value[1]['e'])*value[1]['f'], True)
    sorting['f']    = (lambda value: value[1]['f'], True)
    sorting['e']    = (lambda value: value[1]['e'], False)
    return sorting

"""
Generating dataset from raw file for the first time
"""
def transform_dataset(user_ids, app_names, write=False):
    remove_digits = str.maketrans('', '', digits)
    all_lines = []
    all_data  = []
    users_data = {}
    ctr_uid = 0
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
                split = line.strip().split(',')
                uid = int(split[0])
                act = split[1]
                app = split[2]
                time = int(split[3])    # in ms
                act_int = activity_to_int(act, activities)
                # print(act_int)
                date = datetime.fromtimestamp(time / 1e3)
                app_split = app.translate(remove_digits).replace(':','.').split('.')
                if time != previous_time:
                    app_dist = []
                for i in range(len(app_names)):
                    app_dist.append(0)
                for app_id in app_split:
                    try:
                        idx = app_names.index(app_id)
                        if idx != -1:
                            app_dist[idx] = 1
                    except:
                        ### Because some app names are deleted to save resources
                        pass
                if time != previous_time:
                    soft = (','.join(str(x) for x in app_dist))
                    text = soft + ',' + str(act_int)
                    lines.append(text)
                    ### label is put in the first column
                    data = []
                    data.append(act_int)
                    data.extend(app_dist)
                    all_data.append(data)
                    user_data.append(data)
                    ### Finally update the previous time to match current time
                    previous_time = time
                ctr += 1
                if ctr % 100000 == 0:
                    debug('Processing {:,} lines'.format(ctr), out_file=True)
            debug('len(texts): {}'.format(len(lines)))
            debug('len(file) : {}'.format(ctr))
            if write is True:
                filename = SOFT_FORMAT.format(cd.working_folder, uid)
                remove_file_if_exists(filename)
                write_to_file_buffered(filename, lines, buffer_size=1000)
                del lines[:]
        all_lines.extend(lines)
    debug('Started writing all app and users data into binary files', out_file=True)
    if write is True:
        with open(cd.working_folder + ALL_APP_NAME, 'wb') as f:
            pickle.dump(all_data, f)
        with open(cd.working_folder + USERS_DATA_NAME, 'wb') as f:
            pickle.dump(users_data, f)
    debug('Finished writing all app and users data into binary files', out_file=True)
    return all_lines, all_data, users_data

"""
Generating dataset from cached file (after the first time)
"""
def read_dataset_from_file():
    debug('Started reading dataset from file')
    try:
        with open(cd.working_folder + ALL_APP_NAME, 'rb') as f:
            all_data = pickle.load(f)
    except:
        all_data = []
    try:
        with open(cd.working_folder + USERS_DATA_NAME, 'rb') as f:
            users_data = pickle.load(f)
    except:
        users_data = {}
    debug('Finished reading dataset from file')
    return all_data, users_data

"""
Generating all apps names from raw file for the first time
"""
def get_all_apps(user_ids, stop_words, write=False, split=True):
    remove_digits = str.maketrans('', '', digits)
    app_names = defaultdict(int)
    debug('Starting get all app names')
    for uid in user_ids:
        filename = SOFT_FORMAT.format(cd.dataset_folder, uid)
        try:
            with open(filename) as fr:
                debug(filename, callerid=get_function_name())
                for line in fr:
                    split = line.strip().split(',')
                    app = split[2]
                    if split is True:
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
    if write is True:
        filename = cd.working_folder + APP_NAMES
        remove_file_if_exists(filename)
        texts = []
        for k, v in app_names.items():
            texts.append('{},{}'.format(k,v))
        write_to_file_buffered(filename, texts)
    return app_names

"""
Generating all apps names from cached file (after the first time)
"""
def get_all_apps_buffered(stop_words):
    filename = cd.working_folder + APP_NAMES
    app_names = []
    with open(filename) as fr:
        for line in fr:
            split = line.strip().split(',')
            try:
                f = int(split[1])
                if f > APP_F_THRESHOLD:
                    app_names.append(split[0])
            except:
                pass
    return app_names

### label is put in the first column
def testing(dataset, uid, cached=True):
    debug('Processing: {}'.format(uid))
    dataset = np.array(dataset)
    clfs = classifier_list()
    # print(dataset.shape)
    ncol = dataset.shape[1]
    X = dataset[:,1:ncol] # Remove index 0 
    y = dataset[:,0]
    texts = []
    info = {}
    info['uid'] = uid
    for name, clf in clfs.items():
        debug(name)
        info['clf_name'] = name
        output = evaluation(X, y, clf, info=info, cached=cached)
        acc = output['acc']
        time = output['time']
        text = '{},{},{},{}'.format(uid, name, acc, time)
        texts.append(text)
    return texts

### Generating testing report per user
def generate_testing_report_single(users_data, user_ids, clear_data=False):
    # ### Test
    debug('Evaluating application data (single)', out_file=True)
    output = []
    output.append(COLUMN_NAMES)
    ctr_uid = 0
    for uid, data in users_data.items():
        ctr_uid += 1
        debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(users_data)), out_file=True)
        debug('#Rows: {}'.format(len(data)), out_file=True)
        if uid not in user_ids:
            continue
        result = testing(data, uid)
        output.extend(result)
    if clear_data is True:
        try:
            users_data.clear()
        except Exception as ex:
            debug(ex)
    filename = '{}soft_report_single_{}.csv'.format(cd.working_folder, date.today())
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, output)
    debug(output)

### Generating testing report for all users
def generate_testing_report_agg(all_data, clear_data=False):
    debug('Evaluating application data (single)', out_file=True)
    debug('#Rows: {}'.format(len(all_data)), out_file=True)
    output = []
    output.append(COLUMN_NAMES)
    output.extend(testing(all_data, 'ALL'))
    if clear_data is True:
        try:
            del all_data[:]
            del all_data
        except Exception as ex:
            debug(ex)
    filename = '{}soft_report_agg_{}.csv'.format(cd.working_folder, date.today())
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, output)
    debug(output)

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    debug('--- Program Started ---', out_file=True)
    user_ids  = init()
    ### Stop words for specific app names
    stop_words = init_stop_words(STOP_FILENAME)
    ### Transform original input into training and testing dataset
    app_names = get_all_apps_buffered(stop_words)
    debug('len(app_names): {}'.format(len(app_names)))
    ### Read dataset for the experiments
    # all_data, users_data = read_dataset_from_file()
    # debug('Finished transforming all data: {} users'.format(len(users_data)), out_file=True)
    ### Generate testing report using machine learning evaluation
    # generate_testing_report_single(users_data, user_ids, clear_data=True)
    # generate_testing_report_agg(all_data, clear_data=False)

    # ### Test
    # debug('Evaluating application data')
    # output = []
    # for uid, data in users_data.items():
    #     debug('User: {}'.format(uid), out_file=True)
    #     debug('#Rows: {}'.format(len(data)), out_file=True)
    #     result = testing(data, uid)
    #     output.extend(result)
    # try:
    #     users_data.clear()
    # except Exception as ex:
    #     debug(ex)
    # debug('All data', out_file=True)
    # debug('#Rows: {}'.format(len(all_data)), out_file=True)
    # result = testing(all_data, 'ALL')
    # output.extend(result)
    # try:
    #     del all_data[:]
    #     del all_data
    # except Exception as ex:
    #     debug(ex)
    # filename = '{}soft_report_{}.csv'.format(cd.working_folder, date.today())
    # remove_file_if_exists(filename)
    # write_to_file_buffered(filename, output)
    # debug(output)

    ### Init sorting mechanism
    # sorting = init_sorting_schemes()
    ### Read software files
    # apps_agg, apps_single = read_soft_file_agg(user_ids)
    # debug('len(apps_agg): {}'.format(len(apps_agg)))
    # debug('len(apps_single): {}'.format(len(apps_single)))
    # debug('Activities: {}'.format(activities))

    # app_stats, acts_app = app_statistics(stop_words, activities, apps_single)
    # top_k_apps = select_top_k_apps(TOP_K, activities, acts_app)
    # for i in range(len(activities)):
    #     debug(activities[i], clean=True)
    #     top = top_k_apps[i]
    #     for app_id, value in top:
    #         debug('{},{},{}'.format(app_id, value['e'], value['f']), clean=True)
    #     print()
    ### Create a tuple for software view model by transforming raw data
    ### {frequency, entropy, entropy_frequency}
    debug('--- Program Finished ---', out_file=True)