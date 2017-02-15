"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0
2016/11/20 03:57PM
"""
from general import *
from evaluation import *
import config_directory as cd
import config_variable as var
import numpy as np

SOFT_FORMAT = '{}/{}_soft.csv'
TEMP_FORMAT = '{}/{}_temp.csv'

HOURLY = 24
DAILY  = 7
WEEKLY = 7 * 24

def read_activity_time(uid, cached=False, timeslots=HOURLY):
    act_time = init_activity_time(timeslots)
    if cached:
        filename = TEMP_FORMAT.format(cd.temporal_folder, uid)
        if is_file_exists(filename):
            with open(filename) as fr:
                for line in fr:
                    split = line.strip().split(',')
                    dow = int(split[0])
                    tod = int(split[1])
                    tow = int(split[2])
                    act_int = int(split[3])
                    if timeslots == HOURLY:
                        act_time[act_int][tod] += 1
                    elif timeslots == DAILY:
                        act_time[act_int][dow] += 1
                    elif timeslots == WEEKLY:
                        act_time[act_int][tow] += 1
            return act_time
    ### Otherwise
    filename = SOFT_FORMAT.format(cd.dataset_folder, uid)
    texts = []
    soft_texts = []
    with open(filename) as fr:
        debug(filename, callerid=get_function_name())
        for line in fr:
            split = line.strip().split(',')
            act = split[1]
            time = int(split[3])    # in ms
            act_int = activity_to_int(act, var.activities)
            date = datetime.fromtimestamp(time / 1e3)
            tod = date.hour
            dow = date.weekday()
            tow = date.weekday()*HOURLY + date.hour
            texts.append('{},{},{},{}'.format(dow,tod,tow,act_int))
            if act_int != -1:
                soft_texts.append(line.strip())
                if timeslots == HOURLY:
                    act_time[act_int][tod] += 1
                elif timeslots == DAILY:
                    act_time[act_int][dow] += 1
                elif timeslots == WEEKLY:
                    act_time[act_int][tow] += 1
    remove_file_if_exists(TEMP_FORMAT.format(cd.temporal_folder, uid))
    write_to_file_buffered(TEMP_FORMAT.format(cd.temporal_folder, uid), texts)
    return act_time

def init_activity_time(timeslots=HOURLY):
    activity_hour = []
    for i in range(len(var.activities)):
        a = []
        for i in range(timeslots):
            a.append(0)
        activity_hour.append(a)
    return np.array(activity_hour)

def testing_temporal(uid='ALL'):
    filename = cd.temporal_folder + '/{}_temp.csv'.format(uid)
    

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    user_ids  = init()
    ### Extracting time distribution
    # for uid in user_ids:
    #     debug(uid)
    #     activity_hour = read_activity_time(uid, cached=True, timeslots=DAILY)
    testing_temporal()