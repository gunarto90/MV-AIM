"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.5
2016/12/08 04:30PM
"""
import os
import math
import json
import inspect
from datetime import datetime
from datetime import date
from math import radians, cos, sin, asin, sqrt, pow, exp

import config_directory as cd
import config_variable as var

IS_DEBUG = True

def get_function_name():
    return inspect.stack()[1][3]

def get_parent_name():
    return inspect.stack()[2][3]

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
        return True
    except OSError as exception:
        return False

def is_file_exists(filename):
    try:
        with open(filename, 'r'):
            return True
    except:
        return False

def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def write_to_file(filename, text, append=True, add_linefeed=True):
    if append is True:
        mode = 'a'
    else:
        mode = 'w'
    linefeed = ''
    if add_linefeed is True:
        linefeed = '\n'
    with open(filename, mode) as fw:
        fw.write(str(text) + linefeed)

def write_to_file_buffered(filename, text_list, append=True, buffer_size=10000):
    debug('Writing file: {}'.format(filename))
    counter = 0
    temp_str = ""
    for text in text_list:
        if counter <= buffer_size:
            temp_str = temp_str + text + '\n'
        else:
            write_to_file(filename, temp_str, append, add_linefeed=False)
            temp_str = ""
            counter = 0
        counter += 1
    # Write remaining text
    if temp_str != "":
        write_to_file(filename, temp_str, append, add_linefeed=False)

def debug(message, callerid=None, clean=False, out_stdio=True, out_file=False):
    make_sure_path_exists(cd.log_folder)
    debug_filename = '{}/log_{}.txt'.format(cd.log_folder, date.today())
    if IS_DEBUG == False:
        return
    text = ''
    if clean is False:
        if callerid is None:
            text = '[DEBUG] [{1}] {0}'.format(message, datetime.now())
        else :
            text = '[DEBUG] [{2}] <Caller: {1}> {0}'.format(message, callerid, datetime.now())
    else:
        if callerid is None:
            text = message
        else :
            text = '{0} <Caller: {1}>'.format(message, callerid)
    if out_stdio is True:
        print(text)
    if out_file is True:
        write_to_file(debug_filename, text)
    return text

def activity_to_int(act, activities):
    idx = 0
    for a in activities:
        if a == act:
            return idx
        idx += 1
    # debug(act, get_function_name())
    return -1

def entropy(data, basis=2):
    total = 0.0
    ent = 0
    for item in data:
        total += item
    if total == 0:
        return -1  # No entropy
    for item in data:
        pi = float(item)/total
        if pi == 0:
            continue
        ent -= pi * math.log(pi, basis)
    return ent

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    distance = km * 1000
    return distance # in meter

def init(file='users.json'):
    ### Init user ids
    user_ids = None
    try:
        with open(file) as data_file:
            load = json.load(data_file)
            user_ids = load['users']
    except Exception as ex:
        debug(ex, callerid='init - json')
    ### Init directories
    make_sure_path_exists(cd.working_folder)

    make_sure_path_exists(cd.log_folder)
    make_sure_path_exists(cd.model_folder)
    make_sure_path_exists(cd.report_folder)

    make_sure_path_exists(cd.software_folder)
    make_sure_path_exists(cd.spatial_folder)
    make_sure_path_exists(cd.temporal_folder)

    make_sure_path_exists(cd.soft_cv_model_folder)
    make_sure_path_exists(cd.soft_statistics_folder)
    ### Return user ids
    return user_ids