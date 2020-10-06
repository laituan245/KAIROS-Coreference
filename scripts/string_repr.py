import os
import time
import torch
import random
import copy

from os.path import dirname
from argparse import ArgumentParser

# Helper Function
def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum[0]

def read_event_types(fp):
    types = {}
    with open(fp, 'r') as f:
        for line in f:
            es = line.split('\t')
            type_name = es[1] + '.' + es[3] + '.' + es[5]
            template = es[8]
            unfiltered_args = es[9:]
            args, arg_ctx = {}, 1
            for i in range(0, len(unfiltered_args), 3):
                if len(unfiltered_args[i].strip()) == 0: continue
                args[unfiltered_args[i]] = '<arg{}>'.format(arg_ctx)
                arg_ctx += 1
            types[type_name] = {
                'type_name': type_name,
                'template': template,
                'args': args
            }
    return types

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--input_entity', default='/shared/nas/data/m1/tuanml2/kairos/output/corrected_format/entity.cs')
    parser.add_argument('--input_event', default='/shared/nas/data/m1/tuanml2/kairos/output/corrected_format/event.cs')
    parser.add_argument('--output_entity', default='/shared/nas/data/m1/tuanml2/kairos/output/final/entity.cs')
    parser.add_argument('--output_event', default='/shared/nas/data/m1/tuanml2/kairos/output/final/event.cs')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # DEBUG Mode
    if args.debug:
        args.input_entity = 'resources/samples/output/entity.cs'
        args.input_event = 'resources/samples/output/event.cs'
        args.output_entity = 'resources/samples/final/entity.cs'
        args.output_event = 'resources/samples/final/event.cs'

    # Determine most representative mention for each entity cluster
    entity2canonical, entity2nominal, entity2pronominal, entity2mention = {}, {}, {}, {}
    with open(args.input_entity, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            es = line.split('\t')
            if len(es) <= 3: continue
            entity_id = es[0]
            if not entity_id in entity2canonical: entity2canonical[entity_id] = []
            if not entity_id in entity2nominal: entity2nominal[entity_id] = []
            if not entity_id in entity2pronominal: entity2pronominal[entity_id] = []
            if not entity_id in entity2mention: entity2mention[entity_id] = []
            if es[1] == 'canonical_mention':
                entity2canonical[entity_id].append(es[2][1:-1])
            if es[1] == 'nominal_mention':
                entity2nominal[entity_id].append(es[2][1:-1])
            if es[1] == 'pronominal_mention':
                entity2pronominal[entity_id].append(es[2][1:-1])
            if es[1] == 'mention':
                entity2mention[entity_id].append(es[2][1:-1])
    entity2repr = {}
    for entity in entity2canonical:
        if len(entity2canonical) > 0:
            entity2repr[entity] = find_majority(entity2canonical[entity])
        elif len(entity2nominal) > 0:
            entity2repr[entity] = find_majority(entity2nominal[entity])
        elif len(entity2pronominal) > 0:
            entity2repr[entity] = find_majority(entity2pronominal[entity])
        else:
            entity2repr[entity] = find_majority(entity2mention[entity])

    # Read event_types
    event_types = read_event_types('resources/event_types.tsv')

    # Read input_event file
    event2type, event2args = {}, {}
    with open(args.input_event, 'r', encoding='utf-8') as f:
        for line in f:
            es = line.strip().split('\t')
            if len(es) <= 3:
                if es[1] == 'type':
                    event2type[es[0]] = es[-1]
                continue
            if not (es[1].startswith('mention') or es[1].startswith('canonical_mention')):
                event_type = event2type[es[0]]
                if event_type in event_types: # Consider only events in the KAIROS ontology
                    event_args = event_types[event_type]['args']
                    arg_name = es[1].split('.')[-2].split('_')[-1]
                    arg_nb = event_args[arg_name]
                    if not es[0] in event2args: event2args[es[0]] = {}
                    if not arg_nb in event2args[es[0]]: event2args[es[0]][arg_nb] = []
                    if len(entity2repr[es[2]].strip()) > 0:
                        event2args[es[0]][arg_nb].append(entity2repr[es[2]].strip())

    # Find majority for each args
    for e in event2args:
        event_args = event2args[e]
        for a in event_args:
            event_args[a] = find_majority(event_args[a])

    # Output new entity.cs and event.cs
    dir1 = dirname(args.output_entity)
    dir2 = dirname(args.output_event)
    if not os.path.exists(dir1): os.makedirs(dir1)
    if not os.path.exists(dir2): os.makedirs(dir2)
    with open(args.input_entity, 'r', encoding='utf-8') as input_f:
        with open(args.output_entity, 'w+', encoding='utf-8') as output_f:
            for line in input_f:
                line = line.strip()
                es = line.split('\t')
                if es[1] != 'canonical_mention':
                    output_f.write('{}\n'.format(line))
                else:
                    es[2] = '"{}"'.format(entity2repr[es[0]])
                    line = '\t'.join(es)
                    output_f.write('{}\n'.format(line))

    with open(args.input_event, 'r', encoding='utf-8') as input_f:
        with open(args.output_event, 'w+', encoding='utf-8') as output_f:
            for line in input_f:
                line = line.strip()
                es = line.split('\t')
                event_type = event2type[es[0]]
                if (not es[1].startswith('canonical_mention')) \
                or (not es[0] in event2args):
                    output_f.write('{}\n'.format(line))
                else:
                    event_repr = copy.deepcopy(event_types[event_type]['template'])
                    event_args = event2args[es[0]]
                    for a in event_args:
                        event_repr = event_repr.replace(a, event_args[a])
                    es[2] = '"{}"'.format(event_repr)
                    line = '\t'.join(es)
                    output_f.write('{}\n'.format(line))
