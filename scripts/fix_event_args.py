import os
import time
import torch
import random

from transformers import *
from os.path import dirname

# Helper Functions
def read_cs(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines

def fix_event_args(event_output):
    event_lines = read_cs(event_output)

    # Build event2type
    event2type = {}
    for line in event_lines:
        es = line.strip().split('\t')
        if es[1] == 'type':
            event2type[es[0]] = es[2]

    # Fix argument
    fixed_lines = []
    for line in event_lines:
        es = line.split('\t')
        if len(es) > 3 and (not 'mention' in es[1]):
            type = event2type[es[0]]
            if not es[1].startswith(type):
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

    with open(event_output, 'w+', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write('{}\n'.format(line.strip()))
