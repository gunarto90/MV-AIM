"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.3
2016/11/24 05:39PM
"""
import getopt
import sys
import re
import os
import json
import numpy as np
import setting as st
from datetime import datetime, date
from string import digits
from collections import defaultdict
from general import *
from evaluation import *

SOFT_FORMAT = '{}/{}_soft.csv'
APP_NAMES   = 'app_names.csv'
APP_F_THRESHOLD = 1000
TOP_K = 5
SORTING = 'ef'  ### ef: entropy frequency, f: frequency, e: entropy

def read_soft_file_agg(dataset_folder, user_ids):
    apps_agg = []
    apps_single = {}
    remove_digits = str.maketrans('', '', digits)
    
    for uid in user_ids:
        filename = SOFT_FORMAT.format(dataset_folder, uid)
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

def transform_dataset(dataset_folder, working_folder, user_ids, app_names, write=False):
    remove_digits = str.maketrans('', '', digits)
    all_lines = []
    all_data  = []
    users_data = {}
    for uid in user_ids:
        lines = []
        if write is False:
            user_data = []
            users_data[uid] = user_data
        filename = SOFT_FORMAT.format(dataset_folder, uid)
        debug('Transforming : {}'.format(filename))
        ctr = 0
        with open(filename) as fr:
            # debug(filename, callerid=get_function_name())
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
                app_dist = []
                for i in range(len(app_names)):
                    app_dist.append(0)
                for app_id in app_split:
                    idx = app_names.index(app_id)
                    if idx != -1:
                        app_dist[idx] = 1
                soft = (','.join(str(x) for x in app_dist))
                text = soft + ',' + str(act_int)
                lines.append(text)
                ### label is put in the first column
                if write is False:
                    data = []
                    data.append(act_int)
                    data.extend(app_dist)
                    all_data.append(data)
                    user_data.append(data)
                ctr += 1
                if ctr % 100000 == 0:
                    debug('Processing {} lines'.format(ctr))
            debug(len(lines))
            if write is True:
                filename = SOFT_FORMAT.format(working_folder, uid)
                remove_file_if_exists(filename)
                write_to_file_buffered(filename, lines, buffer_size=1000)
                del lines[:]
        if write is False:
            all_lines.extend(lines)
    return all_lines, all_data, users_data

def read_dataset_from_file(working_folder, user_ids):
    all_data  = []
    users_data = {}
    for uid in user_ids:
        filename = SOFT_FORMAT.format(dataset_folder, uid)
        debug('Transforming : {}'.format(filename))
        ctr = 0
        with open(filename) as fr:
            for line in fr:
                split = line.strip().split(',')
                act_int = int(split[len(split-1)])
                app_dist = int(split[0:len(split-2)])

def get_all_apps(dataset_folder, user_ids, stop_words, working_folder, write=False):
    remove_digits = str.maketrans('', '', digits)
    app_names = defaultdict(int)
    debug('Starting get all app names')
    for uid in user_ids:
        filename = SOFT_FORMAT.format(dataset_folder, uid)
        with open(filename) as fr:
            debug(filename, callerid=get_function_name())
            for line in fr:
                split = line.strip().split(',')
                app = split[2]
                app_split = app.translate(remove_digits).replace(':','.').split('.')
                for app_id in app_split:
                    app_names[app_id] += 1
    for app in stop_words:
        app_names.pop(app, None)
    debug('Finished get all app names')
    if write is True:
        filename = working_folder + APP_NAMES
        remove_file_if_exists(filename)
        texts = []
        for k, v in app_names.items():
            texts.append('{},{}'.format(k,v))
        write_to_file_buffered(filename, texts)
    return app_names

def get_all_apps_buffered(working_folder, stop_words):
    filename = working_folder + APP_NAMES
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
def testing(dataset, uid):
    dataset = np.array(dataset)
    clfs = classifier_list()
    # print(dataset.shape)
    ncol = dataset.shape[1]
    X = dataset[:,1:ncol] # Remove index 0 
    y = dataset[:,0]
    texts = []
    for name, clf in clfs.items():
        # debug(name)
        output = evaluation(X, y, clf)
        acc = output['acc']
        time = output['time']
        text = '{},{},{},{}'.format(uid, name, acc, time)
        texts.append(text)
    return texts

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    debug('--- Program Started ---', out_file=True)
    data, user_ids  = init()
    dataset_folder  = data[st.get_dataset_folder()]
    working_folder  = data[st.get_working_folder()]
    activities      = data[st.get_activities()]
    ### Stop words for specific app names
    stop_app_filename = data[st.get_app_stop()]
    stop_words = init_stop_words(stop_app_filename)
    ### Transform original input into training and testing dataset
    app_names = get_all_apps(dataset_folder, user_ids, stop_words, working_folder, write=True)
    print(len(app_names))
    app_names = get_all_apps_buffered(working_folder, stop_words)
    print(len(app_names))
    # lines, all_data, users_data = transform_dataset(dataset_folder, working_folder, user_ids, app_names, write=True)
    # debug('Finished transforming all data', out_file=True)
    # Free some memory
    # # print(len(lines))
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
    # filename = '{}soft_report_{}.csv'.format(working_folder, date.today())
    # remove_file_if_exists(filename)
    # write_to_file_buffered(filename, output)
    # print(output)

    ### Init sorting mechanism
    # sorting = init_sorting_schemes()
    ### Read software files
    # apps_agg, apps_single = read_soft_file_agg(dataset_folder, user_ids)
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