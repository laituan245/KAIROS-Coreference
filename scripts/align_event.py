import os
import time
import torch
import random

from os.path import dirname
from argparse import ArgumentParser

def align_event(new_input_entity, new_input_event):
    output_path = dirname(new_input_event) + 'temp_event.cs'

    # Read output entity file
    mid2eid = {}
    with open(new_input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            es = line.split('\t')
            if es[1].endswith('mention') or es[1] == 'UNK':
                mid2eid[es[-2]] = es[0]

    # Output file
    with open(output_path, 'w+', encoding='utf-8') as output_f:
        with open(new_input_event, 'r', encoding='utf-8') as event_f:
            for line in event_f:
                line = line.strip()
                if len(line) == 0: continue
                es = line.split('\t')
                if es[1].startswith('type') or es[1].startswith('mention.actual') or es[1].startswith('canonical_mention.actual'):
                    output_f.write('{}\n'.format(line))
                else:
                    mid = es[-2]
                    es[-3] = mid2eid[mid]
                    output_f.write('{}\n'.format('\t'.join(es)))

    # Delete old file
    os.remove(new_input_event)
    os.rename(output_path, new_input_event)
