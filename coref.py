import os
import gc
import json
import time
import torch
import random

from constants import *
from os.path import dirname, join
from argparse import ArgumentParser
from entity_coref import entity_coref
from event_coref import event_coref
from utils import create_dir_if_not_exist, flatten
from scripts import merge_inputs, separate_files, apply_attrs
from scripts import align_relation, align_event, docs_filtering, string_repr, fix_event_types, fix_event_args

# Main code
def main_coref(oneie_output, linking_output, coreference_output, keep_distractors):
    # Attribtes Classifiers
    en_event_cs_path = join(oneie_output, 'en/oneie/m1_m2/cs/event.cs')
    en_json_dir = join(oneie_output, 'en/oneie/m1_m2/json/')

    # Merging en and es
    merged_input = join(coreference_output, 'merged_input')
    create_dir_if_not_exist(coreference_output)

    # Run document filtering
    en_json_dir = join(oneie_output, 'en/oneie/m1_m2/json/')
    es_json_dir = join(oneie_output, 'es/oneie/m1_m2/json/')
    en_filtered_doc_ids, en_distracted_doc_ids = docs_filtering(en_json_dir, language='en')
    es_filtered_doc_ids, es_distracted_doc_ids = docs_filtering(es_json_dir, language='es')
    filtered_doc_ids = en_filtered_doc_ids.union(es_filtered_doc_ids)
    distracted_doc_ids = en_distracted_doc_ids.union(es_distracted_doc_ids)
    output_distractors = join(coreference_output, 'distrators.txt')
    with open(output_distractors, 'w+') as f:
        for _id in distracted_doc_ids:
            f.write('{}\n'.format(json.dumps([_id])))

    # Create a merged input folder
    event_cs, entity_cs, relation_cs, json_dir, linking_output, english_docs, spanish_docs = \
        merge_inputs(oneie_output, linking_output, merged_input)

    # Run document clustering
    clusters = [
        ['L0C04AT6B', 'L0C04AT6J', 'L0C04AT6W', 'L0C04AT6Y', 'L0C04CJ13'], # Chipotle
        ['L0C04D4DA', 'L0C04D4DB', 'L0C04D4DS', 'L0C04D4RB'],              # Peanut Butter Outbreak
    ]
    output_cluster = join(coreference_output, 'clusters.txt')
    with open(output_cluster, 'w+') as f:
        for c in clusters:
            f.write('{}\n'.format(json.dumps(c)))

    # Run entity coref (English)
    output_entity =  join(coreference_output, 'entity.cs')
    filtered_eng_doc_ids = [f for f in filtered_doc_ids if f in english_docs]
    english_entity_pairs = \
        entity_coref(entity_cs, json_dir, linking_output, output_entity, 'en', filtered_eng_doc_ids, clusters, english_docs, spanish_docs)
    gc.collect()

    # Run event coref (English)
    output_event = join(coreference_output, 'event.cs')
    event_coref(event_cs, json_dir, output_event, entity_cs, output_entity, filtered_doc_ids, clusters, english_docs, spanish_docs)
    gc.collect()

    # Run aligning relation
    output_relation = join(coreference_output, 'relation.cs')
    align_relation(entity_cs, output_entity, relation_cs, output_relation)

    # Run aligning event
    align_event(output_entity, output_event)

    # Run string_repr
    string_repr(output_entity, output_event, english_docs)


    # Fix event types
    fix_event_types(output_event)

    # Fix event arguments
    fix_event_args(output_event)

    # Separate files into English / Spanish
    separate_files(output_entity, output_event, output_relation, english_docs, spanish_docs)
