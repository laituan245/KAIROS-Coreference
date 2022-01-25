import os
import glob
import json
import requests

from os.path import isfile, isdir, join
from argparse import ArgumentParser

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def read_data(json_file):
    with open(json_file, 'r') as f:
        data = f.read()
    return data

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--base_input_dir', default='/shared/nas/data/m1/tuanml2/kairos_dryrun2')
    args = parser.parse_args()

    # Process each sub input directory
    while True:
        for dirname in os.listdir(args.base_input_dir):
            print(f'Processing: {dirname}')
            input_dir_path = join(args.base_input_dir, dirname)
            input_fp = join(input_dir_path, 'coref_input.json')
            if not isfile(input_fp): continue
            if isdir(join(input_dir_path, 'all')): continue
            input_data = read_data(input_fp)
            input_data = json.loads(input_data)
            # Call the API
            response = requests.post('http://localhost:20202/process', json={'data': input_data})
            with open(join(input_dir_path, 'sample_response.txt'), 'w+') as f:
                f.write(response.text)
            # convert
            BASE_PATH = input_dir_path
            EN_BASE_PATH = join(BASE_PATH, 'en')
            ES_BASE_PATH = join(BASE_PATH, 'es')
            ALL_BASE_PATH = join(BASE_PATH, 'all')
            create_dir_if_not_exist(ALL_BASE_PATH)
            create_dir_if_not_exist(EN_BASE_PATH)
            create_dir_if_not_exist(ES_BASE_PATH)

            # Read sample_response.txt
            with open(join(input_dir_path, 'sample_response.txt')) as f:
                data = json.loads(f.read())

            # Write files (All)
            with open(join(ALL_BASE_PATH, 'entity.cs'), 'w+') as f:
                f.write(data['entity.cs'])
            with open(join(ALL_BASE_PATH, 'event.cs'), 'w+') as f:
                f.write(data['event.cs'])
            with open(join(ALL_BASE_PATH, 'clusters.txt'), 'w+') as f:
                f.write(data['clusters.txt'])
            with open(join(ALL_BASE_PATH, 'distractors.txt'), 'w+') as f:
                f.write(data['distrators.txt'])

            # Write files (EN)
            with open(join(EN_BASE_PATH, 'entity.cs'), 'w+') as f:
                f.write(data['en']['entity.cs'])
            with open(join(EN_BASE_PATH, 'event.cs'), 'w+') as f:
                f.write(data['en']['event.cs'])

            # Write files (ES)
            with open(join(ES_BASE_PATH, 'entity.cs'), 'w+') as f:
                f.write(data['es']['entity.cs'])
            with open(join(ES_BASE_PATH, 'event.cs'), 'w+') as f:
                f.write(data['es']['event.cs'])
