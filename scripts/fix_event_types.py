import os
import time
import torch
import random
import shutil

from os.path import dirname, join
from utils import create_dir_if_not_exist

# Helper Functions
def read_cs(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum[0]

# Decide types
def decide_representative_type(types):
    detailed_types = []
    for type in types:
        if not 'unspecified' in type.lower():
            detailed_types.append(type)

    repr_detailed_type = find_majority(detailed_types)
    if detailed_types.count(repr_detailed_type) > 1: return repr_detailed_type
    return find_majority(types)


def fix_event_types(event_output):
    event_lines = read_cs(event_output)
    event2types = {}
    for line in event_lines:
        es = line.strip().split('\t')
        if not es[0] in event2types: event2types[es[0]] = []
        if len(es) <= 3 and es[1] == 'type':
            event2types[es[0]].append(es[2].strip())

    event2reprtype = {}
    for e in event2types:
        event2reprtype[e] = decide_representative_type(event2types[e])

    with open(event_output, 'w+', encoding='utf-8') as f:
        for line in event_lines:
            line = line.strip()
            es = line.split('\t')
            if len(es) <= 3 and es[1] == 'type':
                es[2] = event2reprtype[es[0]]
                line = '\t'.join(es)
                f.write('{}\n'.format(line))
            else:
                f.write('{}\n'.format(line))
