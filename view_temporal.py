"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0
2016/11/20 03:57PM
"""
from general import *
import numpy as np

SOFT_FORMAT = '{}/{}_soft.csv'

def read_activity_hour(dataset_folder, uid):
    filename = SOFT_FORMAT.format(dataset_folder, uid)
    activity_hour = init_activity_hour()
    with open(filename) as fr:
        debug(filename, callerid=get_function_name())
        for line in fr:
            split = line.strip().split(',')
            act = split[1]
            time = int(split[3])    # in ms
            act_int = activity_to_int(act, activities)
            date = datetime.fromtimestamp(time / 1e3)
            activity_hour[act_int][date.hour] += 1
    # debug(activity_hour.shape)
    debug(activity_hour.T.shape)
    # for i in range(len(activity_hour.T)):
    #     ent = entropy(activity_hour.T[i])
    #     debug('{},{}'.format(i, ent), clean=True)
    # for i in range(len(activities)):
    #     debug(activity_hour[i], clean=True)
    return activity_hour

def init_activity_hour():
    activity_hour = []
    for i in range(len(activities)):
        a = []
        for i in range(24):
            a.append(0)
        activity_hour.append(a)
    return np.array(activity_hour)

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    data = init()
    dataset_folder  = data[st.get_dataset_folder()]
    working_folder  = data[st.get_working_folder()]
    user_ids        = data[st.get_uids()]
    activities      = data[st.get_activities()]
    ### Extracting time distribution
    for uid in user_ids:
        activity_hour = read_activity_hour(dataset_folder, uid)