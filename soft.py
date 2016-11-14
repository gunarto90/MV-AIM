import getopt
import sys
import re
import os
import json
import setting as st
from string import digits
from general import *

def init(file='setting.json'):
    global dataset_folder
    global working_folder
    global user_ids
    global activities
    try:
        with open(file) as data_file:
            data = json.load(data_file)
            ### Extracting variables
            dataset_folder = data[st.get_dataset_folder()]
            working_folder = data[st.get_working_folder()]
            user_ids = data[st.get_uids()]
            activities = data[st.get_activities()]
    except Exception as ex:
        debug(ex, callerid='init - json')

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
    init()
    debug(dataset_folder)
    debug(user_ids)
    apps_agg = []
    apps_single = {}
    remove_digits = str.maketrans('', '', digits)
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
                app_split = app.translate(remove_digits).replace(':','.').split('.')
                for app_id in app_split:
                    apps_agg.append(app_id)
                    found = apps_single.get(app_id)
                    if found is None:
                        found = []
                    found.append(activity_to_int(act, activities))
                    apps_single[app_id] = found
    # debug(len(apps_agg))
    # debug(len(apps_single))
    # debug(activities)
    acts = []
    for i in range(len(activities)):
        acts.append(0)
    # debug(acts)
    for app_id, uacts in apps_single.items():
        # debug(app_id)
        # debug(len(uacts))
        for a in uacts:
            acts[a] += 1
        # debug(len(acts))
        # debug(entropy(acts))
        text = '{},{},{}'.format(app_id, len(uacts), entropy(acts))
        debug(text, clean=True)
        ### Clean array in every loop
        del acts[:]
        for i in range(len(activities)):
            acts.append(0)
        # break