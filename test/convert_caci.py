import os
import json
import argparse

from os.path import join
from utils import create_dir_if_not_exist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_nb', default=1)
    args = parser.parse_args()

    BASE_PATH = 'test/caci/'
    ALL_BASE_PATH = join(BASE_PATH, 'all{}'.format(args.test_nb))
    JSON_DIR = join(ALL_BASE_PATH, 'json')
    create_dir_if_not_exist(ALL_BASE_PATH)
    create_dir_if_not_exist(JSON_DIR)

    # Read sample_response.txt
    with open(join(BASE_PATH, 'ce100{}.json'.format(args.test_nb))) as f:
        data = json.loads(f.read())
        coref_data = data['coref']
        oneie_data = data['oneie']
        oneie_data_en = oneie_data['en']['json']
        oneie_data_es = oneie_data['es']['json']
        print(oneie_data_en.keys())

    # Write files (All)
    with open(join(ALL_BASE_PATH, 'entity.cs'), 'w+') as f:
        f.write(coref_data['entity.cs'])
    with open(join(ALL_BASE_PATH, 'event.cs'), 'w+') as f:
        f.write(coref_data['event.cs'])
    with open(join(ALL_BASE_PATH, 'clusters.txt'), 'w+') as f:
        f.write(coref_data['clusters.txt'])
    with open(join(ALL_BASE_PATH, 'distractors.txt'), 'w+') as f:
        f.write(coref_data['distrators.txt'])

    # Create json files
    oneie_data = {}
    for key in oneie_data_en: oneie_data[key] = oneie_data_en[key]
    for key in oneie_data_es: oneie_data[key] = oneie_data_es[key]
    for key in oneie_data:
        fn = '{}.json'.format(key)
        fp = join(JSON_DIR, fn)
        with open(fp, 'w+') as f:
            f.write(oneie_data[key])
