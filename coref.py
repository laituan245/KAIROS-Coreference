import os
import json
import time
import torch
import random

from constants import *
from os.path import dirname, join
from argparse import ArgumentParser
from entity_coref import entity_coref
from event_coref import event_coref
from utils import create_dir_if_not_exist
from refine_entity_coref import refine_entity_coref
from attribute_classifiers import generate_hedge_preds, generate_realis_preds, generate_polarity_preds
from scripts import filter_relation, merge_inputs, remove_entities, separate_files
from scripts import align_relation, align_event, docs_filtering, string_repr, fix_event_types

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--oneie_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/')
    parser.add_argument('--linking_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/')
    parser.add_argument('--coreference_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/cross_lingual_coref')
    parser.add_argument('--keep_distractors', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        args.oneie_output = 'resources/quizlet4/'
        args.linking_output = 'resources/quizlet4/'
        args.coreference_output = 'resources/quizlet4/cross_lingual_coref'

    # Attribtes Classifiers
    en_event_cs_path = join(args.oneie_output, 'en/oneie/m1_m2/cs/event.cs')
    en_json_dir = join(args.oneie_output, 'en/oneie/m1_m2/json/')
    generate_hedge_preds(en_event_cs_path, en_json_dir, args.coreference_output)
    generate_realis_preds(en_event_cs_path, en_json_dir, args.coreference_output)
    generate_polarity_preds(en_event_cs_path, en_json_dir, args.coreference_output)
    torch.cuda.empty_cache()

    # Merging en and es
    args.merged_input = join(args.coreference_output, 'merged_input')
    create_dir_if_not_exist(args.coreference_output)

    # Create a merged input folder
    event_cs, entity_cs, relation_cs, json_dir, linking_output, english_docs, spanish_docs = \
        merge_inputs(args.oneie_output, args.linking_output, args.merged_input)

    # Run document filtering
    filtered_doc_ids, distracted_doc_ids = docs_filtering(event_cs, json_dir)
    output_distractors = join(args.coreference_output, 'distrators.txt')
    with open(output_distractors, 'w+') as f:
        for _id in distracted_doc_ids:
            f.write('{}\n'.format(json.dumps([_id])))

    # Run document clustering
    # Cluster 1 ~ 2018 Caracas drone attack | Cluster 2 ~ Utah High School backpack bombing
    clusters = [['K0C041NI3', 'K0C047Z5C', 'K0C041NI5', 'K0C047Z5A', 'K0C041O37', 'K0C041O3D'],
                ['K0C041NHV', 'K0C041NI2', 'K0C041NHW', 'K0C041NHY', 'K0C041NI0', 'K0C047Z59', 'K0C047Z57']]
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

    # Run entity coref (English)
    output_entity =  join(args.coreference_output, 'entity.cs')
    filtered_eng_doc_ids = [f for f in filtered_doc_ids if f in english_docs]
    english_entity_pairs = \
        entity_coref(entity_cs, json_dir, linking_output, output_entity, 'en', filtered_eng_doc_ids, clusters, english_docs, spanish_docs)

    # Run entity coref (Spanish)
    output_entity =  join(args.coreference_output, 'entity.cs')
    filtered_spanish_doc_ids = [f for f in filtered_doc_ids if f in spanish_docs]
    mono_entity_pairs = \
        entity_coref(entity_cs, json_dir, linking_output, output_entity, 'es', filtered_spanish_doc_ids, clusters, english_docs, spanish_docs, predicted_pairs=english_entity_pairs)

    # Run cross-lingual coref
    entity_coref(entity_cs, json_dir, linking_output, output_entity, 'cross', filtered_doc_ids, clusters, english_docs, spanish_docs, predicted_pairs=mono_entity_pairs)

    # The loop stops when refinement process does not modify entity coref anymore
    while True:
        # Run event coref
        output_event = join(args.coreference_output, 'event.cs')
        event_coref(event_cs, json_dir, output_event, entity_cs, output_entity, filtered_doc_ids, clusters)

        # Run aligning relation
        output_relation = join(args.coreference_output, 'relation.cs')
        align_relation(entity_cs, output_entity, relation_cs, output_relation)

        # Run aligning event
        align_event(output_entity, output_event)

        # Run string_repr
        string_repr(output_entity, output_event)

        # Run filter_relation
        filter_relation(output_event, output_relation)

        print('refinement')
        changed = refine_entity_coref(output_entity, output_event)
        print('changed = {}'.format(changed))
        if not changed:
            break

    # Remove non-participating-entities
    remove_entities(output_entity, output_event, output_relation)

    # Fix event types
    fix_event_types(output_event)

    # Separate files into English / Spanish
    separate_files(output_entity, output_event, output_relation, english_docs, spanish_docs)
