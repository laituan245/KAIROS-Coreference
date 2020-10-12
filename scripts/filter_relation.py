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
def filter_relation(new_input_event, new_input_relation):
    # Read event file
    event_args = set()
    with open(new_input_event, 'r', encoding='utf-8') as event_f:
        for line in event_f:
            line = line.strip()
            if len(line) == 0: continue
            es = line.split('\t')
            if es[1].startswith('type') or es[1].startswith('mention.actual') or es[1].startswith('canonical_mention.actual'):
                pass
            else:
                event_args.add(es[2])

    # Read relation file
    filtered_lines, skipped = [], 0
    with open(new_input_relation, 'r', encoding='utf-8') as relation_f:
        for line in relation_f:
            line = line.strip()
            if len(line) == 0: continue
            es = line.split('\t')
            arg_0, arg_1 = es[0], es[2]
            if arg_0 in event_args or arg_1 in event_args:
                filtered_lines.append(line)
            else:
                skipped += 1

    # Output new relation file
    with open(new_input_relation, 'w+', encoding='utf-8') as relation_f:
        for line in filtered_lines:
            relation_f.write('{}\n'.format(line))
    print('Remove {} relations'.format(skipped))
