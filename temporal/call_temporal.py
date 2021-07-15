import os
import glob
import json
import requests

from os.path import join

def read_data(json_file):
    with open(json_file, 'r') as f:
        data = f.read()
    return data

input_data = read_data('temporal_input.json')
input_data = json.loads(input_data)

response = requests.post('http://localhost:8000/temporal_pipeline', json=input_data)
with open('temporal_response.txt', 'w+') as f:
    f.write(response.text)
