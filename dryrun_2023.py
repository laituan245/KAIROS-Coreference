import json
import requests
from os.path import join

ce_name = 'ce2013'
base_path = 'resources/dryrun2023'
sample_input_fp = join(base_path, '{}.json'.format(ce_name))
sample_response_fp = join(base_path, '{}_response.txt'.format(ce_name))

with open(sample_input_fp, 'r') as f:
    input_data = json.loads(f.read())

response = requests.post('http://localhost:20202/process', json={'data': input_data})
with open(sample_response_fp, 'w+') as f:
    f.write(response.text)

