import os
import time
import torch
import copy
import random

from transformers import *
from os.path import dirname
from utils import read_event_types

# Helper Functions
def read_cs(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines

def fix_event_args(event_output):
    event_lines = read_cs(event_output)
    event_types = read_event_types('resources/event_types.tsv')

    # Build event2type
    event2type = {}
    for line in event_lines:
        es = line.strip().split('\t')
        if es[1] == 'type':
            event2type[es[0]] = es[2]

    # [Stage 1] Fix argument
    fixed_lines, fixed_stage_1 = [], 0
    for line in event_lines:
        es = line.split('\t')
        if len(es) > 3 and (not 'mention' in es[1]):
            type = event2type[es[0]]
            if not es[1].startswith(type):
                fixed_stage_1 += 1
                old_type = es[1].split('_')[0]
                arg_name = es[1].split('.')[-2].split('_')[-1]
                if type == 'Conflict.Attack.DetonateExplode':
                    if 'Conflict.Attack' in old_type:
                        if arg_name in ['Target', 'Instrument', 'Place', 'Attacker']:
                            line = line.replace(old_type, type)
                    if 'Life.Die' in old_type:
                        if arg_name == 'Victim':
                            line = line.replace(old_type, type)
                            line = line.replace('Victim', 'Target')
                elif type == 'Movement.Transportation.Evacuation':
                    if 'Movement.Transportation' in old_type:
                        if arg_name in ['Transporter', 'PassengerArtifact', 'Vehicle', 'Origin', 'Destination']:
                            line = line.replace(old_type, type)
                elif type == 'Contact.Contact.Broadcast':
                        if 'Contact' in old_type:
                            if arg_name in ['Participant']:
                                line = line.replace(old_type, type)
                else:
                    line = line.replace(old_type, type)
        fixed_lines.append(line)
    assert(len(fixed_lines) == len(event_lines))
    print('[Stage 1] Number of fixed lines: {}'.format(fixed_stage_1))

    # [Stage 2]
    event_lines = copy.deepcopy(fixed_lines)
    fixed_lines, stage2_discarded = [], 0
    for line in event_lines:
        es = line.split('\t')
        if len(es) > 3 and (not 'mention' in es[1]):
            type = event2type[es[0]]
            arg_name = es[1].split('.')[-2].split('_')[-1]
            event_args = event_types[type]['args']
            if not arg_name in event_args:
                stage2_discarded += 1
                continue
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    print('stage2_discarded = {}'.format(stage2_discarded))

    with open(event_output, 'w+', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write('{}\n'.format(line.strip()))
