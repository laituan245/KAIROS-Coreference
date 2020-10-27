import os
import time
import torch
import random
import shutil

from os.path import dirname, join
from utils import create_dir_if_not_exist

def read_cs(path, skip_first_twolines=False):
    with open(path, 'r', encoding='utf-8') as f:
        if skip_first_twolines:
            f.readline()
            f.readline()
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines

def fix_cs(path, language):
    result_lines = []
    lines = read_cs(path)
    for line in lines:
        es = line.split('\t')
        for i in range(len(es)):
            if es[i].startswith(':Entity_EDL_'):
                id_part = es[i].split('_')[-1]
                es[i] = ':Entity_EDL_{}_{}'.format(language.upper(), id_part)
            elif es[i].startswith(':Event_'):
                id_part = es[i].split('_')[-1]
                es[i] = ':Event_{}_{}'.format(language.upper(), id_part)
        result_lines.append('\t'.join(es))
    return result_lines

def copy_files(source_dir, target_dir):
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.copyfile(join(source_dir, file_name), join(target_dir, file_name))


def merge_inputs(oneie_output, linking_output, merged_input):
    create_dir_if_not_exist(merged_input)

    # Determine english docs and spanish docs
    english_docs = os.listdir(join(oneie_output, 'en/oneie/m1_m2/json/'))
    english_docs = [f[:f.rfind('.json')] for f in english_docs if f.endswith('.json')]

    spanish_docs = os.listdir(join(oneie_output, 'es/oneie/m1_m2/json/'))
    spanish_docs = [f[:f.rfind('.json')] for f in spanish_docs if f.endswith('.json')]

    # Merge entity.cs files
    en_oneie_entity = fix_cs(join(oneie_output, 'en/oneie/m1_m2/cs/entity.cs'), 'ENG')
    es_oneie_entity = fix_cs(join(oneie_output, 'es/oneie/m1_m2/cs/entity.cs'), 'SPA')
    all_oneie_entity = en_oneie_entity + es_oneie_entity
    entity_cs = join(merged_input, 'entity.cs')
    with open(entity_cs, 'w+', encoding='utf-8') as f:
        for line in all_oneie_entity:
            f.write('{}\n'.format(line))

    # Merge event.cs files
    en_oneie_event = fix_cs(join(oneie_output, 'en/oneie/m1_m2/cs/event.cs'), 'ENG')
    es_oneie_event = fix_cs(join(oneie_output, 'es/oneie/m1_m2/cs/event.cs'), 'SPA')
    all_oneie_event = en_oneie_event + es_oneie_event
    event_cs = join(merged_input, 'event.cs')
    with open(event_cs, 'w+', encoding='utf-8') as f:
        for line in all_oneie_event:
            f.write('{}\n'.format(line))

    # Merge relation.cs files
    en_oneie_relation = fix_cs(join(oneie_output, 'en/oneie/m1_m2/cs/relation.cs'), 'ENG')
    es_oneie_relation = fix_cs(join(oneie_output, 'es/oneie/m1_m2/cs/relation.cs'), 'SPA')
    all_oneie_relation = en_oneie_relation + es_oneie_relation
    relation_cs = join(merged_input, 'relation.cs')
    with open(relation_cs, 'w+', encoding='utf-8') as f:
        for line in all_oneie_relation:
            f.write('{}\n'.format(line))

    # Merge entity link
    en_entity_linking = read_cs(join(linking_output, 'en/linking/en.linking.wikidata.cs'), skip_first_twolines=True)
    es_entity_linking = read_cs(join(linking_output, 'es/linking/es.linking.wikidata.cs'), skip_first_twolines=True)
    all_entity_linking = en_entity_linking + es_entity_linking
    new_linking_output = join(merged_input, 'linking.wikidata.cs')
    with open(new_linking_output, 'w+', encoding='utf-8') as f:
        f.write('RPI_BLENDER1')
        f.write('\n')
        for line in all_entity_linking:
            f.write('{}\n'.format(line))

    # Merge json dir
    json_dir = join(merged_input, 'json/')
    create_dir_if_not_exist(json_dir)
    copy_files(join(oneie_output, 'en/oneie/m1_m2/json/'), json_dir)
    copy_files(join(oneie_output, 'es/oneie/m1_m2/json/'), json_dir)

    return event_cs, entity_cs, relation_cs, json_dir, new_linking_output, english_docs, spanish_docs
