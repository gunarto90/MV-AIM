import getopt
import sys
import re
import os
import json
import setting as st
from general import *

def init(file='setting.json'):
    global dataset_folder
    global working_folder
    with open(file) as data_file:    
        data = json.load(data_file)
        dataset_folder = data[st.get_dataset_folder()]
        working_folder = data[st.get_working_folder()]

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