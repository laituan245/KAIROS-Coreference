import os
import uuid
import logging
import shutil
import argparse
import time
import json

from os.path import join
from shutil import rmtree
from flask import Flask, request, jsonify

TMP_DIR = None
LANGUAGES = ['en', 'es']
app = Flask(__name__)

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

def parse_cs_data(data, output_file):
    lines = data.splitlines()
    with open(output_file, 'w+') as f:
        for line in lines:
            f.write('{}\n'.format(line.strip()))

@app.route('/process', methods=['POST'])
def process():
    form = request.get_json()
    data = form.get('data')

    # Create tmp dir
    run_tmp_dir = os.path.join(TMP_DIR, str(uuid.uuid4()))
    os.makedirs(run_tmp_dir, exist_ok=True)
    logger.info('Created tmp output directory: {}'.format(run_tmp_dir))

    # Create input folder
    for lang in LANGUAGES:
        input_lang_dir = os.path.join(run_tmp_dir, lang)
        os.makedirs(input_lang_dir, exist_ok=True)
        oneie_data = json.loads(data['oneie'][lang])['oneie']
        edl_data = json.loads(data['edl'][lang])
        # oneie cs folder
        oneie_dir = os.path.join(input_lang_dir, 'oneie/m1_m2')
        os.makedirs(oneie_dir, exist_ok=True)
        os.makedirs(join(oneie_dir,'cs'), exist_ok=True)
        oneie_cs_data = oneie_data['cs']
        for datatype in ['entity', 'event', 'relation']:
            parse_cs_data(oneie_cs_data[datatype], join(join(oneie_dir,'cs'), '{}.cs'.format(datatype)))
        # oneie json folder
        os.makedirs(join(oneie_dir,'json'), exist_ok=True)
        json_dir = join(oneie_dir,'json')
        for doc_id in oneie_data['json']:
            with open(join(json_dir, '{}.json'.format(doc_id)), 'w+') as output_json_dir:
                output_json_dir.write(oneie_data['json'][doc_id])
        # edl folder
        edl_dir = os.path.join(input_lang_dir, 'linking')
        os.makedirs(edl_dir, exist_ok=True)
        # cs file
        edl_cs_filepath = join(edl_dir, '{}.linking.wikidata.cs'.format(lang))
        with open(edl_cs_filepath, 'w+') as output_edl_cs_file:
            output_edl_cs_file.write('{}'.format(edl_data['cs']['entity']))
        # tab file
        edl_tab_filepath = join(edl_dir, '{}.linking.wikidata.tab'.format(lang))
        with open(edl_tab_filepath, 'w+') as output_edl_tab_file:
            output_edl_tab_file.write('{}'.format(edl_data['tab']['entity']))

    # Remove the tmp dir
    #rmtree(run_tmp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_dir', help='Temporary output directory', default='./tmp', required=True)
    parser.add_argument('--port', default=20202)
    args = parser.parse_args()
    TMP_DIR = args.tmp_dir

    logger.info('done.')
    logger.info('start...')
    app.run('0.0.0.0', port=int(args.port))
