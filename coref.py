import os
import json
import time
import torch
import random
import logging

from constants import *
from os.path import dirname, join
from entity_coref import entity_coref
from event_coref import event_coref
from utils import create_dir_if_not_exist, flatten
from scripts import align_relation, align_event, filter_relation, fix_event_args, fix_event_types

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

# Main code
def coref_main(oneie_output, linking_output, coreference_output, clusters):
    create_dir_if_not_exist(coreference_output)

    # Sanity check
    success_file_path = join(coreference_output, '_success')
    if os.path.exists(success_file_path):
        logger.info('[coref] A successful file already exists, exit')

    # Wait for signal from linking
    success_file_path = join(dirname(linking_output), '_success')
    s = time.time()
    while not os.path.exists(success_file_path):
        #print('coref has been waiting for: %.3f seconds' % (time.time()-s))
        logger.info('coref has been waiting for: %.3f seconds' % (time.time()-s))
        time.sleep(15)

    # Dummy distractor txt file
    filtered_doc_ids = set(flatten(clusters))
    output_distractors = join(coreference_output, 'distrators.txt')
    with open(output_distractors, 'w+') as f:
        pass

    # Run document clustering
    output_cluster = join(coreference_output, 'clusters.txt')
    with open(output_cluster, 'w+') as f:
        for c in clusters:
            f.write('{}\n'.format(json.dumps(c)))

    # Run entity coref
    entity_cs = join(oneie_output, 'cs/entity.cs')
    json_dir = join(oneie_output, 'json')
    output_entity =  join(coreference_output, 'entity.cs')
    entity_coref(entity_cs, json_dir, linking_output, output_entity, 'en', filtered_doc_ids, clusters)

    # Run event coref
    event_cs = join(oneie_output, 'cs/event.cs')
    json_dir = join(oneie_output, 'json')
    output_event = join(coreference_output, 'event.cs')
    event_coref(event_cs, json_dir, output_event, 'en', entity_cs, output_entity, filtered_doc_ids, clusters)

    # Run aligning relation
    input_relation = join(oneie_output, 'cs/relation.cs')
    output_relation = join(coreference_output, 'relation.cs')
    align_relation(entity_cs, output_entity, input_relation, output_relation)

    # Run aligning event
    align_event(output_entity, output_event)

    # Run filter_relation
    filter_relation(output_event, output_relation)

    # Fix event types
    fix_event_types(output_event)

    # Fix event arguments
    fix_event_args(output_event)

    # Write a new success file
    success_file_path = join(coreference_output, '_success')
    with open(success_file_path, 'w+') as f:
        f.write('success')
