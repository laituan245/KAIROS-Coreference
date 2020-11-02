import os
import json
import time
import torch
import random
import logging

from constants import *
from flask import Flask
from os.path import dirname, join
from argparse import ArgumentParser
from entity_coref import entity_coref
from event_coref import event_coref
from utils import create_dir_if_not_exist
from refine_entity_coref import refine_entity_coref
from scripts import align_relation, align_event, docs_filtering, string_repr, filter_relation, remove_entities

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
    parser.add_argument('--keep_distractors', action='store_true')
    args = parser.parse_args()
    args.ta = int(args.ta)
    assert(args.language in ['en', 'es'])
    assert(args.ta in [1, 2])

    if args.debug:
        args.oneie_output = 'resources/quizlet4/{}/output/oneie/m1_m2'.format(args.language)
        args.linking_output = 'resources/quizlet4/{}/output/linking/{}.linking.wikidata.cs'.format(args.language, args.language)
        args.coreference_output = 'resources/quizlet4/{}/output/coref/'.format(args.language)

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

    # Run document filtering
    event_cs = join(args.oneie_output, 'cs/event.cs')
    json_dir = join(args.oneie_output, 'json')
    filtered_doc_ids, distracted_doc_ids = docs_filtering(event_cs, json_dir, args.language)
    output_distractors = join(args.coreference_output, 'distrators.txt')
    with open(output_distractors, 'w+') as f:
        for _id in distracted_doc_ids:
            f.write('{}\n'.format(json.dumps([_id])))

    # Run document clustering
    clusters = [list(filtered_doc_ids)]
    output_cluster = join(args.coreference_output, 'clusters.txt')
    with open(output_cluster, 'w+') as f:
        for c in clusters:
            f.write('{}\n'.format(json.dumps(c)))
    if args.keep_distractors:
        for distractor in distracted_doc_ids:
            clusters.append([distractor])

    # Update filtered_doc_ids to contain all docs if keep_distractors
    if args.keep_distractors:
        filtered_doc_ids = filtered_doc_ids.union(distracted_doc_ids)

    # Run entity coref
    entity_cs = join(args.oneie_output, 'cs/entity.cs')
    json_dir = join(args.oneie_output, 'json')
    output_entity =  join(args.coreference_output, 'entity.cs')
    entity_coref(entity_cs, json_dir, args.linking_output, output_entity, args.language, filtered_doc_ids, clusters)

    # The loop stops when refinement process does not modify entity coref anymore
    while True:
        # Run event coref
        event_cs = join(args.oneie_output, 'cs/event.cs')
        json_dir = join(args.oneie_output, 'json')
        output_event = join(args.coreference_output, 'event.cs')
        event_coref(event_cs, json_dir, output_event, args.language, entity_cs, output_entity, filtered_doc_ids, clusters)

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
    remove_entities(output_entity, output_event, output_relation)

    # Write a new success file
    if not args.debug:
        success_file_path = join(args.coreference_output, '_success')
        with open(success_file_path, 'w+') as f:
            f.write('success')

    # At the end
    app.run('0.0.0.0', port=int(args.port))
