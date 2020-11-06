import os
import json
import time
import torch
import random
import logging

from constants import *
from flask import Flask
from data import read_json_docs
from os.path import dirname, join
from argparse import ArgumentParser
from entity_coref import entity_coref
from event_coref import event_coref
from utils import create_dir_if_not_exist
from refine_entity_coref import refine_entity_coref
from scripts import align_relation, align_event, string_repr, filter_relation, remove_entities

ONEIE = 'oneie'
app = Flask(__name__)
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--oneie_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/en/oneie/m1_m2')
    parser.add_argument('--linking_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/en/linking/en.linking.wikidata.cs')
    parser.add_argument('--coreference_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/en/coref/')
    parser.add_argument('--ta', default=1)
    parser.add_argument('--language', default='en')
    parser.add_argument('--port', default=3300)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args.ta = int(args.ta)
    assert(args.language in ['en', 'es'])
    assert(args.ta in [1, 2])

    if args.debug:
        args.oneie_output = 'resources/oneie/m1_m2'
        args.linking_output = 'resources/linking/{}.linking.wikidata.cs'.format(args.language)
        args.coreference_output = 'resources/coref/'

    create_dir_if_not_exist(args.coreference_output)

    # Sanity check
    success_file_path = join(args.coreference_output, '_success')
    if os.path.exists(success_file_path):
        logger.info('[coref] A successful file already exists, exit')

    # Wait for signal from linking
    if not args.debug:
        success_file_path = join(dirname(args.linking_output), '_success')
        s = time.time()
        while not os.path.exists(success_file_path):
            #print('coref has been waiting for: %.3f seconds' % (time.time()-s))
            logger.info('coref has been waiting for: %.3f seconds' % (time.time()-s))
            time.sleep(15)
        os.remove(success_file_path)

    # Run document filtering (DUMMY)
    json_dir = join(args.oneie_output, 'json')
    doc_ids = list(read_json_docs(json_dir).keys())
    with open(join(args.coreference_output, 'distrators.txt'), 'w+') as f:
        # No distractors
        pass

    # Run document clustering (DUMMY - Within-Doc Coref)
    clusters = [[d] for d in doc_ids]
    with open(join(args.coreference_output, 'clusters.txt'), 'w+') as f:
        for c in clusters:
            f.write('{}\n'.format(json.dumps(c)))

    # Run entity coref
    entity_cs = join(args.oneie_output, 'cs/entity.cs')
    json_dir = join(args.oneie_output, 'json')
    output_entity =  join(args.coreference_output, 'entity.cs')
    entity_coref(entity_cs, json_dir, args.linking_output, output_entity, args.language, set(doc_ids), clusters)

    # The loop stops when refinement process does not modify entity coref anymore
    while True:
        # Run event coref
        event_cs = join(args.oneie_output, 'cs/event.cs')
        json_dir = join(args.oneie_output, 'json')
        output_event = join(args.coreference_output, 'event.cs')
        event_coref(event_cs, json_dir, output_event, args.language, entity_cs, output_entity, set(doc_ids), clusters)

        # Run aligning relation
        input_relation = join(args.oneie_output, 'cs/relation.cs')
        output_relation = join(args.coreference_output, 'relation.cs')
        align_relation(entity_cs, output_entity, input_relation, output_relation)

        # Run aligning event
        align_event(output_entity, output_event)

        # Run string_repr
        string_repr(output_entity, output_event)

        # Run filter_relation
        if args.ta == 2:
            filter_relation(output_event, output_relation)

        print('refinement')
        changed = refine_entity_coref(output_entity, output_event)
        print('changed = {}'.format(changed))
        if not changed:
            break

    # Remove non-participating-entities
    #remove_entities(output_entity, output_event, output_relation)

    # Write a new success file
    if not args.debug:
        success_file_path = join(args.coreference_output, '_success')
        with open(success_file_path, 'w+') as f:
            f.write('success')

    # At the end
    app.run('0.0.0.0', port=int(args.port))
