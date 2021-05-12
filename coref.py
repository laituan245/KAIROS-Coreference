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
from refine_entity_coref import refine_entity_coref
from attribute_classifiers import generate_hedge_preds, generate_realis_preds, generate_polarity_preds
from scripts import filter_relation, merge_inputs, remove_entities, separate_files, apply_attrs, remove_arguments
from scripts import align_relation, align_event, docs_filtering, es_translation, string_repr, fix_event_types, fix_event_args

# Main code
def main_coref(oneie_output, linking_output, coreference_output, keep_distractors):
    # Attribtes Classifiers
    en_event_cs_path = join(oneie_output, 'en/oneie/m1_m2/cs/event.cs')
    en_json_dir = join(oneie_output, 'en/oneie/m1_m2/json/')
    generate_hedge_preds(en_event_cs_path, en_json_dir, coreference_output)
    gc.collect()
    generate_realis_preds(en_event_cs_path, en_json_dir, coreference_output)
    gc.collect()
    generate_polarity_preds(en_event_cs_path, en_json_dir, coreference_output)
    gc.collect()
    torch.cuda.empty_cache()

    # Use ES translation

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

    # Remove non-participating-entities (Before main processing)
    #remove_entities(entity_cs, event_cs, relation_cs)

    # Run document clustering
    clusters = [list(filtered_doc_ids)]
    if keep_distractors:
        for distractor in distracted_doc_ids:
            clusters.append([distractor])
    output_cluster = join(coreference_output, 'clusters.txt')
    with open(output_cluster, 'w+') as f:
        for c in clusters:
            f.write('{}\n'.format(json.dumps(c)))

    # Update filtered_doc_ids to contain all docs if keep_distractors
    if keep_distractors:
        filtered_doc_ids = filtered_doc_ids.union(distracted_doc_ids)

    # Run entity coref (English)
    output_entity =  join(coreference_output, 'entity.cs')
    filtered_eng_doc_ids = [f for f in filtered_doc_ids if f in english_docs]
    english_entity_pairs = \
        entity_coref(entity_cs, json_dir, linking_output, output_entity, 'en', filtered_eng_doc_ids, clusters, english_docs, spanish_docs)
    gc.collect()

    # Run entity coref (Spanish)
    output_entity =  join(coreference_output, 'entity.cs')
    filtered_spanish_doc_ids = [f for f in filtered_doc_ids if f in spanish_docs]
    mono_entity_pairs = \
        entity_coref(entity_cs, json_dir, linking_output, output_entity, 'es', filtered_spanish_doc_ids, clusters, english_docs, spanish_docs, predicted_pairs=english_entity_pairs)
    gc.collect()

    # Run cross-lingual coref
    entity_coref(entity_cs, json_dir, linking_output, output_entity, 'cross', filtered_doc_ids, clusters, english_docs, spanish_docs, predicted_pairs=mono_entity_pairs)
    gc.collect()

    # The loop stops when refinement process does not modify entity coref anymore
    while True:
        # Run event coref
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

        # Run filter_relation
        filter_relation(output_event, output_relation)

        print('refinement')
        changed = refine_entity_coref(output_entity, output_event)
        print('changed = {}'.format(changed))
        if not changed:
            break

    # Run remove arguments
    remove_arguments(output_entity, output_event, coreference_output)

    # Remove non-participating-entities
    #remove_entities(output_entity, output_event, output_relation)

    # Fix event types
    fix_event_types(output_event)

    # Fix event arguments
    fix_event_args(output_event)

    # apply_attrs
    apply_attrs(coreference_output)

    # Separate files into English / Spanish
    separate_files(output_entity, output_event, output_relation, english_docs, spanish_docs)
