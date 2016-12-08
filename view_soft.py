"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.12
2016/12/08 04:34PM
"""

"""
## ToDo !!:
- From the users_data --> generate activity X apps vector which shows the score of each apps regarding its {frequency, entropy, entropy-frequency, other}
- Top-k apps for the classifier
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

APP_PART_NAMES          = 'app_names_part.csv'
APP_FULL_NAMES          = 'app_names_full.csv'
CATEGORY_NAME           = 'category_lookup.csv'
APP_CATEGORY            = 'app_category.csv'
STOP_FILENAME           = 'stop_app.txt'

APP_AGG_NAME            = 'app_agg.bin'
APP_SINGLE_NAME         = 'app_single.bin'

APP_STATS_NAME          = 'acts_stats_{}_{}.bin' ### mode uid

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

REPORT_AGG_PART_NAME    = '{}/soft_report_agg_part_{}.csv'
REPORT_AGG_FULL_NAME    = '{}/soft_report_agg_full_{}.csv'
REPORT_AGG_CAT_NAME     = '{}/soft_report_agg_cat_{}.csv'

REPORT_TOD_NAME         = '{}/soft_report_tod_{}_{}.csv'
REPORT_DOW_NAME         = '{}/soft_report_dow_{}_{}.csv'
REPORT_TOW_NAME         = '{}/soft_report_tow_{}_{}.csv'

APP_F_THRESHOLD         = 1000  ## Minimum occurrence of the app throughout the dataset
TIME_WINDOW             = 1000  ## in ms

DEFAULT_SORTING         = 'f'       ### ef: entropy frequency, f: frequency, erf: entropy*sqrt(freqency)
WEIGHTING               = 'g'       ### g: general (unweighted), r: rank on list

COLUMN_NAMES            = 'UID,Classifier,Accuracy,TrainTime(s),TestTime(s)'

"""
@Initialization methods
"""
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
    # sorting['e']    = (lambda value: value[1]['e'], False)
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
            time = 0
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
                        data.append(time)
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
def testing(dataset, uid, cached=True, mode='Default', groups=None):
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
            output = evaluation(X, y, clf, info=info, cached=cached, mode=mode, groups=groups)
            acc = output['acc']
            time_train = output['time_train']
            time_test = output['time_test']
            text = '{},{},{},{},{}'.format(uid, name, acc, time_train, time_test)
            texts.append(text)
        return texts
    except Exception as ex:
        debug('Error on testing', ex)
        return None

### Generating testing report per user
def generate_testing_report_single(users_data, user_ids, clear_data=False, full=False, categories=None, cached=True):
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
        # debug(data)
        result = testing(data, uid, cached=cached, mode=mode)
        if result is not None:
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
def generate_testing_report_agg(users_data, user_ids, clear_data=False, full=False, categories=None, cached=True):
    debug('Evaluating application data (agg)', out_file=True)
    output = []
    output.append(COLUMN_NAMES)
    ctr_uid = 0
    if categories is not None:
        mode = 'cat'
    elif full:
        mode = 'full'
    else:
        mode = 'part'
    dataset = []
    groups  = []
    for uid, data in users_data.items():
        ctr_uid += 1
        if uid not in user_ids:
            continue
        # debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(users_data)), out_file=True)
        # debug('#Rows: {}'.format(len(data)), out_file=True)
        for x in data:
            dataset.append(x)
            groups.append(ctr_uid)
    result = testing(dataset, uid, cached=cached, mode=mode, groups=groups)
    if clear_data:
        try:
            del dataset[:]
        except Exception as ex:
            debug(ex, get_function_name())
    if result is not None:
        output.extend(result)
    if categories is not None:
        filename = REPORT_AGG_CAT_NAME.format(cd.report_folder, date.today())
    elif full:
        filename = REPORT_AGG_FULL_NAME.format(cd.report_folder, date.today())
    else:
        filename = REPORT_AGG_PART_NAME.format(cd.report_folder, date.today())
    remove_file_if_exists(filename)
    write_to_file_buffered(filename, output)

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
        write_to_file_buffered(REPORT_TOD_NAME.format(cd.report_folder, mode, uid), texts)
        del texts[:]
    if day_of_week is not None:
        for app, arr in day_of_week.items():
            if mode.lower() == 'cat':
                text = '{},{}'.format(categories[app], (','.join(str(x) for x in arr)))
            else:
                text = '{},{}'.format(app, (','.join(str(x) for x in arr)))
            texts.append(text)
        write_to_file_buffered(REPORT_DOW_NAME.format(cd.report_folder, mode, uid), texts)
        del texts[:]
    if time_of_week is not None:
        for app, arr in time_of_week.items():
            if mode.lower() == 'cat':
                text = '{},{}'.format(categories[app], (','.join(str(x) for x in arr)))
            else:
                text = '{},{}'.format(app, (','.join(str(x) for x in arr)))
            texts.append(text)
        write_to_file_buffered(REPORT_TOW_NAME.format(cd.report_folder, mode, uid), texts)
        del texts[:]

def extract_app_statistics(data, mode, uid, app_names=None, categories=None, cached=True):
    acts_app = []   ### For every activities it would have a dictionary
    if cached:
        try:
            filename = cd.soft_statistics_folder + APP_STATS_NAME.format(mode, uid)
            with open(filename, 'rb') as f:
                acts_app = pickle.load(f)
        except:
            extract_app_statistics(data, mode, uid, app_names, categories, cached=False)
    else:
        frequencies = []    ### consist of {} -- dict of (app name and frequency score)
        entropies = {}      ### dict of (app name and entropy score)
        dataset = np.array(data)
        n_row = dataset.shape[0]
        n_col = dataset.shape[1]
        data_s = []
        cond_s = []

        names = None
        if categories is not None:
            names = categories
        else:
            names = app_names

        for i in range(len(var.activities)):
            cond_s.append(dataset[:,2] == i)    ### Compare the activity label with current activity
            data_s.append(dataset[cond_s[i]])

        ### Extract basic stats for each app
        fs = []
        for i in range(len(var.activities)):
            X = data_s[i][:,3:n_col] # Remove index 0 (uid), index 1 (time), and index 2 (activities)
            y = data_s[i][:,2]
            f = np.sum(X, axis=0)
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
            # debug(act_dict)

        ### Write acts_app data into file
        filename = cd.soft_statistics_folder + APP_STATS_NAME.format(mode, uid)
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
        top = []
        for app_id, value in sorted(acts_app[i].items(), key=sort, reverse=desc):
            if k >= top_k:
                break
            top.append((app_id, value))
            k += 1
        top_k_apps[i] = top
    return top_k_apps

def evaluate_topk_apps(users_data, user_ids, mode, topk, sorting, sort_mode, app_names=None, categories=None, cached=True, single=True):
    ctr_uid = 0
    if single:
        for uid, data in users_data.items():
            ctr_uid += 1
            debug('User: {} [{}/{}]'.format(uid, ctr_uid, len(users_data)), out_file=True)
            debug('#Rows: {}'.format(len(data)), out_file=True)
            if uid not in user_ids:
                continue
            soft_evaluation(data, uid, mode, topk, sorting, sort_mode, app_names=app_names, categories=categories, cached=cached)
    else:
        dataset = []
        groups  = []
        for uid, data in users_data.items():
            ctr_uid += 1
            if uid not in user_ids:
                continue
            for x in data:
                dataset.append(x)
                groups.append(ctr_uid)
        soft_evaluation(dataset, uid, mode, topk, sorting, sort_mode, app_names=app_names, categories=categories, cached=cached, groups=groups)

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    debug('--- Program Started ---', out_file=True)
    MODE = [
        'Full', 'Part', 'Cat'
    ]  ## 'Full', 'Part', 'Cat', 'Hybrid'

    TOP_K = 10
    SORT_MODE = 'erf'

    ### Init sorting mechanism
    sorting = init_sorting_schemes()
    ### Init user ids
    user_ids  = init()

    ### Stop words for specific app names
    stop_words = init_stop_words(STOP_FILENAME)

    for mode in MODE:
        debug('Mode is : {}'.format(mode))
        if mode.lower() == 'full' or mode.lower() == 'cat':
            full_app_name = True
        else:
            full_app_name = False

        ### Apps categories
        categories = None
        app_cat = None
        if mode.lower() == 'cat':
            categories = init_categories()
            app_cat = init_app_category()

        ### Transform original input into training and testing dataset
        app_names = get_all_apps(user_ids, stop_words, write=True, split=not full_app_name, cached=True)
        debug('len(app_names): {}'.format(len(app_names)))

        ### Read dataset for the experiments
        users_data = transform_dataset(user_ids, app_names, write=True, full=full_app_name, categories=categories, app_cat=app_cat, cached=False)
        debug('Finished transforming all data: {} users'.format(len(users_data)), out_file=True)

        ### Generate testing report using machine learning evaluation
        # generate_testing_report_single(users_data, user_ids, clear_data=False, full=full_app_name, categories=categories, cached=False)
        # generate_testing_report_agg(users_data, user_ids, clear_data=True, full=full_app_name, categories=categories, cached=True)

        ### Extract time of each apps
        # global_timeline, personal_timeline = extract_time_data(user_ids, mode, app_cat=app_cat, cached=False)
        # ### Global timeline
        # time_of_day, day_of_week, time_of_week = time_slots_extraction(global_timeline)
        # timeline_report(mode, 'GLOBAL', categories=categories, time_of_day=time_of_day, day_of_week=day_of_week, time_of_week=time_of_week)
        # ### Personal timeline
        # for uid in user_ids:
        #     time_of_day, day_of_week, time_of_week = time_slots_extraction(personal_timeline[uid])
        #     timeline_report(mode, uid, categories=categories, time_of_day=time_of_day, day_of_week=day_of_week, time_of_week=time_of_week)

        ### Top-k apps evaluation
        # evaluate_topk_apps(users_data, user_ids, mode, TOP_K, sorting, SORT_MODE, app_names=app_names, categories=categories, cached=False, single=True)

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