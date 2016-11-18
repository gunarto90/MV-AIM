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

# Main function
if __name__ == '__main__':
    ### Load options (if any)
    # try:
    #     opts, args = getopt.getopt(sys.argv[1:],"p:k:s:f:m:",["project=","topk=","start=","finish=","mode="])
    # except getopt.GetoptError:
    #     err_msg = 'pgt.py -m MODE -p <0 gowalla / 1 brightkite> -k <top k users> -s <start position>'
    #     debug(err_msg, 'opt error')
    #     sys.exit(2)
    # if len(opts) > 0:
    #     pass
    ### Initialize variables from json file
    dataset_folder, working_folder, user_ids, activities = init()
    debug(dataset_folder)
    debug(user_ids)
    apps_agg = []
    apps_single = {}
    remove_digits = str.maketrans('', '', digits)
    activity_hour = []
    for i in range(len(activities)):
        a = []
        for i in range(24):
            a.append(0)
        activity_hour.append(a)
    for uid in user_ids:
        filename = dataset_folder + str(uid) + '_soft.csv'
        with open(filename) as fr:
            debug(filename, callerid='soft files')
            for line in fr:
                split = line.strip().split(',')
                uid = int(split[0])
                act = split[1]
                app = split[2]
                time = int(split[3])    # in ms
                act_int = activity_to_int(act, activities)
                date = datetime.fromtimestamp(time / 1e3)
                activity_hour[act_int][date.hour] += 1
                app_split = app.translate(remove_digits).replace(':','.').split('.')
                for app_id in app_split:
                    apps_agg.append(app_id)
                    found = apps_single.get(app_id)
                    if found is None:
                        found = []
                    found.append(act_int)
                    apps_single[app_id] = found
    ### Time
    # aa = np.array(activity_hour)
    # debug(aa.shape)
    # debug(aa.T.shape)
    # for i in range(len(aa.T)):
    #     ent = entropy(aa.T[i])
    #     debug('{},{}'.format(i, ent), clean=True)
    # for i in range(len(activities)):
    #     debug(activity_hour[i], clean=True)
    ### Apps
    debug(len(apps_agg))
    debug(len(apps_single))
    debug(activities)
    acts = []
    acts_app = []
    app_stats = {}
    stop_words = ['com', 'google', 'example', 'android', 'htc', 'samsung', 'sony', 'unstable', 'process', 'systemui', 'system', 'nctuhtclogger']
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
        text = '{},{},{},{}'.format(app_id, len(uacts), entropy(acts, len(activities)), acts)
        app_stats[app_id] = (len(uacts), entropy(acts, len(activities)))
        # debug(text, clean=True)
        ### Clean array in every loop
        del acts[:]
        for i in range(len(activities)):
            acts.append(0)
    for app_id, (f, e) in app_stats.items():
        for i in range(len(activities)):
            acts_map = acts_app[i]
            found = acts_map.get(app_id)
            if found is None:
                found = {'f':0, 'e':0.0}
            total = sum(acts_map[x]['f'] for x in acts_map)
            if total > 0:
                found['f'] = float(found['f'])/total
            else:
                found['f'] = 0.0
            found['e'] = e
            if found['f'] > 0 and found['e'] > 0:
                acts_map[app_id] = found
    for i in range(len(activities)):
        topk = 3
        k = 0
        debug(activities[i])
        debug(acts_app[i])
        # ### Sorting by 1-entropy * frequency (descending)
        # for app_id, value in sorted(acts_app[i].items(), key=lambda value: (1-value[1]['e'])*value[1]['f'], reverse=True):
        #     if k >= topk:
        #         break
        #     debug('{},{},{}'.format(app_id, value['e'], value['f']), clean=True)
        #     k += 1
        # debug('---')
        # ### Sorting by frequency (descending)
        # for app_id, value in sorted(acts_app[i].items(), key=lambda value: value[1]['f'], reverse=True):
        #     if k >= topk:
        #         break
        #     # debug('{},{},{}'.format(app_id, value['e'], value['f']), clean=True)
        #     debug(app_id)
        #     debug(value)
        #     k += 1
        # debug('---')