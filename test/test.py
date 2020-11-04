import os
import glob
import json
import requests

def read_data(json_file):
    f = open(json_file, 'r')
    data = f.read()
    f.close()
    return data

input_data = read_data('test/sample_input.json')
input_data = json.loads(input_data)
input_data['oneie']['es'] = input_data['oneie']['en']
input_data['edl']['es'] = input_data['oneie']['es']

response = requests.post('http://localhost:20202/process', json={'data': input_data})
