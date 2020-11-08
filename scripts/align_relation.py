import os
import time
import torch
import random

from os.path import dirname
from argparse import ArgumentParser

# Helper function
def create_dir_if_not_exist(dir):
    if not os.path.exists(dir): os.makedirs(dir)

# Main code
def align_relation(original_input_entity, new_input_entity, input_relation, output_relation):
    create_dir_if_not_exist(dirname(output_relation))

    # Read (old) entity file
    olde2mid = {}
    with open(original_input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            es = line.split('\t')
            if es[1].startswith('mention') or es[1].startswith('canonical_mention'):
                olde2mid[es[0]] = es[-2]

    # Read (new) entity file
    mid2eid = {}
    with open(new_input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            es = line.split('\t')
            if es[1].endswith('mention'):
                mid2eid[es[-2]] = es[0]

    # Output file
    skipped = 0
    with open(output_relation, 'w+', encoding='utf-8') as output_f:
        with open(input_relation, 'r', encoding='utf-8') as event_f:
            for line in event_f:
                line = line.strip()
                if len(line) == 0: continue
                es = line.split('\t')
                assert(len(es) == 5)
                try:
                    es[0] = mid2eid[olde2mid[es[0]]]
                    es[2] = mid2eid[olde2mid[es[2]]]
                    if es[0] == es[2]:
                        skipped += 1
                        continue
                except:
                    skipped += 1
                    continue
                output_f.write('{}\n'.format('\t'.join(es)))
    print('Skipped {} relations'.format(skipped))
