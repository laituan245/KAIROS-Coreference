import os
import glob
import json
import requests

def read_data(json_file):
    f = open(json_file, 'r')
    data = f.read()
    f.close()
    return data

en_edl = read_data('test/data/en_edl.json')
en_oneie = read_data('test/data/en_oneie.json')
es_edl = read_data('test/data/es_edl.json')
es_oneie = read_data('test/data/es_oneie.json')

data = {
    'oneie': {},
    'edl': {}
}

data['oneie']['en'] = en_oneie
data['oneie']['es'] = es_oneie
data['edl']['en'] = en_edl
data['edl']['es'] = es_edl

response = requests.post('http://localhost:20202/process', json={'data': data})
