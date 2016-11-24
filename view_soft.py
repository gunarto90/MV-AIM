"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.2
2016/11/24 04:45PM
"""
import getopt
import sys
import re
import os
import json
import numpy as np
import setting as st
from datetime import datetime
from string import digits
from general import *
from evaluation import *

SOFT_FORMAT = '{}/{}_soft.csv'
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
        user_data = []
        users_data[uid] = user_data
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
                data = []
                data.append(act_int)
                data.extend(app_dist)
                all_data.append(data)
                user_data.append(data)
        if write is True:
            filename = SOFT_FORMAT.format(working_folder, uid)
            remove_file_if_exists(filename)
            write_to_file_buffered(filename, lines)
        all_lines.extend(lines)
    return all_lines, all_data, users_data

def get_all_apps(dataset_folder, user_ids):
    remove_digits = str.maketrans('', '', digits)
    app_names = []
    for uid in user_ids:
        filename = SOFT_FORMAT.format(dataset_folder, uid)
        with open(filename) as fr:
            # debug(filename, callerid=get_function_name())
            for line in fr:
                split = line.strip().split(',')
                app = split[2]
                app_split = app.translate(remove_digits).replace(':','.').split('.')
                for app_id in app_split:
                    if app_id not in app_names:
                        app_names.append(app_id)
    return app_names

### label is put in the first column
def testing(dataset):
    dataset = np.array(dataset)
    clfs = classifier_list()
    # print(dataset.shape)
    ncol = dataset.shape[1]
    X = dataset[:,1:ncol] # Remove index 0 
    y = dataset[:,0]
    for name, clf in clfs.items():
        debug(name)
        output = evaluation(X, y, clf)
        for name, result in output.items():
            # if name != 'y1':
            if name == 'acc':
                debug('{}\t [{}]'.format(result, name))

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    data = init()
    dataset_folder  = data[st.get_dataset_folder()]
    working_folder  = data[st.get_working_folder()]
    user_ids        = data[st.get_uids()]
    activities      = data[st.get_activities()]
    ### Stop words for specific app names
    stop_app_filename = data[st.get_app_stop()]
    stop_words = init_stop_words(stop_app_filename)
    ### Transform original input into training and testing dataset
    app_names = get_all_apps(dataset_folder, user_ids)
    # print(len(app_names))
    lines, all_data, users_data = transform_dataset(dataset_folder, working_folder, user_ids, app_names, write=False)
    # print(len(lines))
    debug(len(all_data))
    ### Test
    for uid, data in users_data.items():
        debug('User: {}'.format(uid))
        testing(data)
    debug('All data')
    testing(all_data)

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