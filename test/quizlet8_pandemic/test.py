import os
import glob
import json
import requests

from os.path import join

BASE_PATH = 'test/quizlet8_pandemic'

def read_data(json_file):
    with open(json_file, 'r') as f:
        data = f.read()
    return data

input_data = read_data(join(BASE_PATH, 'sample_input.json'))
input_data = json.loads(input_data)

response = requests.post('http://localhost:20202/process', json={'data': input_data})
with open(join(BASE_PATH, 'sample_response.txt'), 'w+') as f:
    f.write(response.text)
