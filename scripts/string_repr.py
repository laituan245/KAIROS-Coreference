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

    return maximum

def find_majority_2(kvs, english_docs):
    ks = [a[0] for a in kvs]
    vs = [a[1] for a in kvs]
    majority_k = find_majority(ks)[0]

    majority_vs, majority_vs_english = [], []
    for i in range(len(vs)):
        if ks[i] == majority_k:
            majority_vs.append(vs[i])
            if vs[i].split(':')[0] in english_docs:
                majority_vs_english.append(vs[i])

    if len(majority_vs_english) > 0:
        majority_v = majority_vs_english[0]
    else:
        majority_v = majority_vs[0]

    return majority_k, majority_v

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

def string_repr(new_input_entity, new_input_event, english_docs):
    output_entity = dirname(new_input_entity) + 'entity_2.cs'
    output_event = dirname(new_input_event) + 'event_2.cs'

    # Determine most representative mention for each entity cluster
    entity2canonical, entity2nominal, entity2pronominal, entity2mention = {}, {}, {}, {}
    with open(new_input_entity, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            es = line.split('\t')
            if len(es) <= 4: continue
            entity_id = es[0]
            if not entity_id in entity2canonical: entity2canonical[entity_id] = []
            if not entity_id in entity2nominal: entity2nominal[entity_id] = []
            if not entity_id in entity2pronominal: entity2pronominal[entity_id] = []
            if not entity_id in entity2mention: entity2mention[entity_id] = []
            if es[1] == 'canonical_mention':
                cur_mention = es[2][1:-1].strip()
                cur_loc = es[3].strip()
                if len(cur_mention) == 0: continue
                entity2mention[entity_id].append((cur_mention, cur_loc))
                if cur_mention in ['Teen', 'I', 'He', 'She', 'They', 'It']: continue
                if cur_mention[0].islower(): continue
                entity2canonical[entity_id].append((cur_mention, cur_loc))
            if es[1] == 'nominal_mention':
                cur_mention = es[2][1:-1].strip()
                cur_loc = es[3].strip()
                if len(cur_mention) == 0: continue
                entity2mention[entity_id].append((cur_mention, cur_loc))
                entity2nominal[entity_id].append((cur_mention, cur_loc))
            if es[1] == 'pronominal_mention':
                cur_mention = es[2][1:-1].strip()
                cur_loc = es[3].strip()
                if len(cur_mention) == 0: continue
                entity2mention[entity_id].append((cur_mention, cur_loc))
                entity2pronominal[entity_id].append((cur_mention, cur_loc))
            if es[1] == 'mention':
                cur_mention = es[2][1:-1].strip()
                cur_loc = es[3].strip()
                if len(cur_mention) == 0: continue
                entity2mention[entity_id].append((cur_mention, cur_loc))
    entity2repr = {}
    for entity in entity2canonical:
        if len(entity2canonical[entity]) > 0:
            entity2repr[entity] = find_majority_2(entity2canonical[entity], english_docs)
        elif len(entity2nominal[entity]) > 0:
            entity2repr[entity] = find_majority_2(entity2nominal[entity], english_docs)
        elif len(entity2pronominal[entity]) > 0:
            entity2repr[entity] = find_majority_2(entity2pronominal[entity], english_docs)
        else:
            entity2repr[entity] = find_majority_2(entity2mention[entity], english_docs)

    # Read event_types
    event_types = read_event_types('resources/event_types.tsv')

    # Read new_input_event file
    event2type, event2args = {}, {}
    with open(new_input_event, 'r', encoding='utf-8') as f:
        for line in f:
            es = line.strip().split('\t')
            if len(es) <= 4:
                if es[1] == 'type':
                    event2type[es[0]] = es[-1]
                continue
            if not (es[1].startswith('mention') or es[1].startswith('canonical_mention')):
                event_type = event2type[es[0]]
                if event_type in event_types: # Consider only events in the KAIROS ontology
                    event_args = event_types[event_type]['args']
                    arg_name = es[1].split('.')[-2].split('_')[-1]
                    if not arg_name in event_args: continue
                    arg_nb = event_args[arg_name]
                    if not es[0] in event2args: event2args[es[0]] = {}
                    if not arg_nb in event2args[es[0]]: event2args[es[0]][arg_nb] = []
                    if len(entity2repr[es[2]][0].strip()) > 0:
                        event2args[es[0]][arg_nb].append(entity2repr[es[2]][0].strip())

    # Find majority for each args
    for e in event2args:
        event_args = event2args[e]
        for a in event_args:
            argmax, ctx = find_majority(event_args[a])
            if len(event_args[a]) == 0: continue
            if ctx / len(event_args[a]) > 0.25 and ctx > 1:
                event_args[a] = argmax

    # Output new entity.cs and event.cs
    with open(new_input_entity, 'r', encoding='utf-8') as input_f:
        with open(output_entity, 'w+', encoding='utf-8') as output_f:
            for line in input_f:
                line = line.strip()
                es = line.split('\t')
                if es[1] != 'canonical_mention':
                    output_f.write('{}\n'.format(line))
                else:
                    es[2] = '"{}"'.format(entity2repr[es[0]][0])
                    es[3] = entity2repr[es[0]][1].strip()
                    line = '\t'.join(es)
                    output_f.write('{}\n'.format(line))

    with open(new_input_event, 'r', encoding='utf-8') as input_f:
        with open(output_event, 'w+', encoding='utf-8') as output_f:
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
                        if type(event_args[a]) == list: continue
                        event_repr = event_repr.replace(a, event_args[a])
                    es[2] = '"{}"'.format(event_repr)
                    line = '\t'.join(es)
                    output_f.write('{}\n'.format(line))

    # Delete old file
    os.remove(new_input_event)
    os.remove(new_input_entity)
    os.rename(output_entity, new_input_entity)
    os.rename(output_event, new_input_event)
