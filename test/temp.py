import json
from os.path import join

test_nb = 2
path = 'test/end2end_test{}'.format(test_nb)
sample_input_fp = join(path, 'sample_input.json')
sample_response_fp = join(path, 'sample_response.txt')

with open(sample_input_fp, 'r') as f:
    ce_data = json.loads(f.read())

with open(sample_response_fp, 'r') as f:
    ce_data['coref'] = json.loads(f.read())

with open('test/caci/ce100{}.json'.format(test_nb), 'w+') as f:
    f.write(json.dumps(ce_data))
