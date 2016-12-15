"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0
2016/11/28 01:26PM
"""

# https://play.google.com/store/apps/details?id=com.facebook.katana
import urllib.request
from bs4 import BeautifulSoup
from general import *
import re
import config_directory as cd

IN_FILENAME = 'app_names_fullname.csv'
OUT_FILENAME = 'app_category.csv'
URL_FORMAT = 'https://play.google.com/store/apps/details?id={}&hl=en'

pattern = re.compile('([A-Za-z]+)')

def open_link(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        page = response.read()
        # page = page.decode('unicode-escape')
        page = str(page).encode('utf-8')
        # print(page)
        return page

def parsing_html(page):
    soup = BeautifulSoup(page, 'html.parser')
    # print(soup.prettify())

    find = soup.find('span', attrs={'itemprop':'genre'}, text=True)
    for node in find:
        category = ''.join(node)
    return category

def find_app_category(app_id):
    try:
        page = open_link(URL_FORMAT.format(app_id))
        category = parsing_html(page)
        print(app_id, category)
    except Exception as ex:
        # debug(app_id, ex)
        category = '#Internal'
    return category

def main():
    apps = []
    with open(cd.working_folder + IN_FILENAME) as f:
        for line in f:
            split = line.split(',')
            apps.append(split[0].strip())
    texts = []
    for app_id in apps:
        category = find_app_category(app_id)
        text = '{},{}'.format(app_id, category)
        texts.append(text)
    write_to_file_buffered(cd.working_folder + OUT_FILENAME, texts)

if __name__ == "__main__":
    main()
