import os
import time
import torch
import random
import shutil

from os.path import dirname, join
from utils import create_dir_if_not_exist

def read_cs(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines

def extract_entities(lines):
    entities = set()
    for line in lines:
        es = line.strip().split('\t')
        for e in es:
            if e.startswith('Entity_EDL'):
                entities.add(e)
    return entities

def remove_entities(entity_output, event_output, relation_output):
    events_entities = extract_entities(read_cs(event_output))
    relations_entities = extract_entities(read_cs(relation_output))
    participating_entities = events_entities.union(relations_entities)

    removed = set()
    all_entities_lines = read_cs(entity_output)
    with open(entity_output, 'w+') as f:
        for line in all_entities_lines:
            cur_entity = line.strip().split('\t')[0]
            if not cur_entity in participating_entities:
                removed.add(cur_entity)
                continue
            f.write('{}\n'.format(line))
    print('removed {}'.format(len(removed)))
