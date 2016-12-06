"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.1
2016/12/06 04:57PM
"""
#!/usr/bin/env python

# Custom import
import json
import foursquare

import config_directory as cd
import config_variable as var

from general import *
from evaluation import *

ST_FORMAT       = '{}/{}_gps.csv'          ## Original gps data
SEARCH_RADIUS   = 500

class Output:
    def __init__(self, _lat, _lon, _cat_ids):
        self.lat = _lat
        self.lon = _lon
        self.cat_ids = _cat_ids

    def __str__(self):
        return '{},{},{}'.format(self.lat, self.lon, self.cat_ids)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

# Foursquare related
def load_secret(config_secret):
    with open(config_secret, 'r') as cred:
        json_str = cred.read()
        json_data = json.loads(json_str)
        client_id = json_data['client_id']
        client_secret = json_data['client_secret']
    debug('Configuration loaded')
    return client_id, client_secret

def auth_4sq(client_id, client_secret):
    # Construct the client object
    client = foursquare.Foursquare(client_id=client_id, client_secret=client_secret)
    # Build the authorization url for your app
    auth_uri = client.oauth.auth_url()
    return client

def search_venue_categories(client, lat, lon, search_radius):
    cat_ids = []
    ll = str(lat) + ',' + str(lon)
    try:
        b = client.venues.search(params={'intent':'browse', 'll': ll, 'radius':search_radius, 'limit':50})
        vv = b['venues']
        for v in vv:
            cats = v['categories']
            for cat in cats:
                cat_id = cat['id']
                cat_ids.append(cat_id)
    except Exception as ex:
        debug('Search Venue Categories Exception : {0}'.format(ex))
        write_to_file('error.log', '[{}]Search Venue Categories Exception : {}\n'.format(str(datetime.now()), ex))
        if str(ex) == 'Quota exceeded':
            print('Let the program sleep for 10 minutes')
            time.sleep(600) # Delay 10 minutes for another crawler
            return None
    return cat_ids

def init_time_location(f_time_location):
    ### Init location data
    user_ids  = init()
    coordinates = []
    for uid in user_ids:
        filename = ST_FORMAT.format(cd.dataset_folder, uid)
        debug(filename)
        try:
            with open(filename) as f:
                for line in f:
                    split = line.strip().split(',')
                    timestamp = int(split[2])
                    lat = float(split[3])
                    lon = float(split[4])
                    coordinates.append((timestamp, lat, lon))
        except Exception as ex:
            # debug(ex, 'Open ST file')
            pass
    texts = []
    for (timestamp, lat, lon) in coordinates:
        text = '{},{:4f},{:4f}'.format(timestamp, lat, lon)
        texts.append(text)
    remove_file_if_exists(f_time_location)
    write_to_file_buffered(f_time_location, texts)

def filter_location(f_time_location, f_location):
    coor = []
    param = -2
    debug('read started')
    with open(f_time_location) as fr:
        for line in fr:
            split = line.strip().split(',')
            lat = split[1]
            lon = split[2]

            text = '{},{}'.format(lat[:param], lon[:param])
            if text not in coor:
                coor.append(text)
    debug('read finished')
    write_to_file_buffered(f_location, text, buffer_size=1000)

def obj_dict(obj):
    return obj.__dict__

def init_categories(f_categories):
    ## Need to capture high level location's functionality
    pass

# Main function
if __name__ == '__main__':
    ### Initialize foursquare API
    f_secret = 'config_secret.json'
    f_time_location = cd.spatial_folder + 'time_location.csv'
    f_location = cd.spatial_folder + 'location.csv'
    # init_time_location(f_time_location)
    # filter_location(f_time_location, f_location)
    coordinates = []
    with open(f_location) as f:
        for line in f:
            split = line.strip().split(',')
            lat = float(split[0])
            lon = float(split[1])
            coordinates.append((lat, lon))

    # debug(coordinates)

    # ### Foursquare API
    client_id, client_secret = load_secret(f_secret)  # Load config_secret.json for credential
    client = auth_4sq(client_id, client_secret)

    output = []
    i = 0
    while i  < len(coordinates):
        (lat, lon) = coordinates[i]
        cat_ids = search_venue_categories(client, lat, lon, SEARCH_RADIUS)
        if cat_ids is None:
            debug('Error while processing ({},{})'.format(lat, lon))
            debug('Processed {} of {} coordinates [{}%]'.format(i, len(coordinates), float(i)*100/len(coordinates)))
            continue
        # debug(cat_ids)
        output.append(Output(lat, lon, cat_ids))
        i += 1
        if i % 100 == 0:
            debug('Processed {} of {} coordinates [{}%]'.format(i, len(coordinates), float(i)*100/len(coordinates)))
    f_out_json = cd.spatial_folder + 'cat_ids.json'
    with open(f_out_json, 'w') as fw:
        json.dump(output, fw, default=obj_dict)