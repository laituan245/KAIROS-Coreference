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

response = requests.post('http://localhost:20202/process', json={'data': input_data})
with open('test/sample_response.txt', 'w+') as f:
    f.write(response.text)
