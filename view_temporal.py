"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0
2016/11/20 03:57PM
"""
from general import *
import evaluation as eva
import config_directory as cd
import config_variable as var
import numpy as np
import time
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

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

def testing_temporal(user_ids, uid='ALL', method='original', cached=True):
    dataset = []
    groups = []
    ctr_uid = 0
    for xuid in user_ids:
        ctr_uid += 1
        if uid != 'ALL' and int(xuid) != int(uid):
            continue
        filename = cd.temporal_folder + '/{}_temp.csv'.format(xuid)
        debug('Read from {}'.format(filename))
        data = read_csv(filename)
        for x in data:
            dataset.append(x)
            groups.append(ctr_uid)
    dataset = np.array(dataset)
    ncol = dataset.shape[1]
    X = dataset[:,0:1] # 0 Day of Week, 1 Time of Day, 2 Time of Week
    y = dataset[:,ncol-1]
    if uid == 'ALL':
        cv, n_split = eva.get_cv(5, groups, X, y)
    else:
        cv, n_split = eva.get_cv(5, None, X, y)
    clfs = eva.classifier_list()
    debug('Finished splitting cross validation')
    texts = []
    for clf_name, clf in clfs.items():
        accs = []
        train_time = 0.0
        test_time = 0.0
        mean_acc = 0.0
        i = 0
        pickle_name = cd.temp_cache + '/{}_{}.bin'.format(uid, clf_name)
        for (train, test) in cv:
            if method != 'original':
                Xt, yt = eva.sampling(X[train], y[train], method=method)
            else:
                Xt, yt = X[train], y[train]
            success = True
            load = False
            query_time = time.time()
            # if cached:
            #     try:
            #         if pickle_name is None:
            #             raise Exception('Filename is None')
            #         with open(pickle_name, 'rb') as f:
            #             fit = pickle.load(f)
            #             load = True
            #             debug('Loaded {}'.format(pickle_name))
            #     except:
            #         success = False
            # try:
            #     if not cached or not success:
            #         fit = clf.fit(Xt, yt)
            # except Exception as ex:
            #     debug(ex, get_function_name())
            fit = clf.fit(Xt, yt)
            train_time += (time.time() - query_time)

            query_time = time.time()
            inference = fit.predict(X[test])
            test_time += (time.time() - query_time)
            acc = accuracy_score(y[test], inference)
            accs.append(acc)
            # try:
            #     if filename is not None and not load:
            #         with open(pickle_name, 'wb') as f:
            #             pickle.dump(fit, f)
            #             debug('Writing to {}'.format(pickle_name))
            # except Exception as ex:
            #     debug(ex, get_function_name())
            mean_acc += acc
            debug('[{}] [{}] Accuracy [{} of {}] : {}'.format(uid, clf_name, i+1, n_split, acc))
            i += 1
        mean_acc /= n_split
        texts.append('{},{},{},{}'.format(clf_name,mean_acc,train_time,test_time))
    write_to_file_buffered('{}/report_{}.csv'.format(cd.temp_report, uid), texts)

def plot_heatmap(data, xlabel=None, xtick=True):
    # http://seaborn.pydata.org/generated/seaborn.heatmap.html
    ax = sns.heatmap(data, yticklabels=var.activities_short, xticklabels=xtick)
    ax.set_ylabel('Activities')
    ax.set_xlabel(xlabel)
    plt.show()

def heatmap(user_ids, uid='ALL'):
    dataset = []
    ctr_uid = 0
    filename = cd.temporal_folder + '/{}_temp.csv'.format(uid)
    debug('Read from {}'.format(filename))
    dataset = read_csv(filename)
    ### Time vs Activity
    dow_dist = []
    tod_dist = []
    tow_dist = []    
    for i in range(len(var.activities)):
        dow_dist.append([])
        tod_dist.append([])
        tow_dist.append([])        
        for x in range(DAILY):
            dow_dist[i].append(0)
        for x in range(HOURLY):
            tod_dist[i].append(0)
        for x in range(WEEKLY):
            tow_dist[i].append(0)
    for data in dataset:
        act = int(data[3])
        dow_dist[act][int(data[0])] += 1
        tod_dist[act][int(data[1])] += 1
        tow_dist[act][int(data[2])] += 1
    debug(dow_dist)
    ### Plots
    plot_heatmap(dow_dist, 'Day of Week', var.DAY_OF_WEEK)
    plot_heatmap(tod_dist, 'Time of Day')
    plot_heatmap(tow_dist, 'Time of Week', False)

# Main function
if __name__ == '__main__':
    ### Initialize variables from json file
    user_ids  = init()
    ### Extracting time distribution
    # for uid in user_ids:
    #     debug(uid)
    #     activity_hour = read_activity_time(uid, cached=True, timeslots=DAILY)
    testing_temporal(user_ids, uid='ALL')
    # heatmap(user_ids, uid='ALL')

"""
1. Make a heatmap
a. Time vs Activities (DoW/ToD/ToW)
b. Apps vs Time (Full/Cat/Hybrid) vs (DoW/ToD/ToW)
c. Apps vs Activities (Full/Cat/Hybrid)
"""