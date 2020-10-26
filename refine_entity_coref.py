import os
import time
import torch
import random

from constants import *
from collections import Counter
from utils import flatten, read_event_types
from algorithms import UndirectedGraph

ARGS = ['<arg1>', '<arg2>', '<arg3>', '<arg4>', '<arg5>']

def refine_entity_coref(new_input_entity, new_input_event):
    event_types = read_event_types('resources/event_types.tsv')

    # Read new_input_event
    mid2eid, entity2mentions, entity2type, entity2link, mid2line = {}, {}, {}, {}, {}
    with open(new_input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            original_line = line
            es = line.strip().split('\t')
            if es[1].endswith('mention'):
                mid2eid[es[-2]] = es[0]
                if not es[0] in entity2mentions: entity2mentions[es[0]] = set()
                entity2mentions[es[0]].add(es[-2])
                # Update mid2line
                if not es[-2] in mid2line: mid2line[es[-2]] = []
                mid2line[es[-2]].append(original_line)
            elif es[1] == 'type':
                entity2type[es[0]] = es[2]
            elif es[1] == 'link':
                entity2link[es[0]] = es[2]

    # Read new_input_event
    event2type, event2args = {}, {}
    with open(new_input_event, 'r', encoding='utf-8') as f:
        for line in f:
            es = line.strip().split('\t')
            if len(es) <= 4:
                if es[1] == 'type':
                    event2type[es[0]] = es[2]
                continue
            if not (es[1].startswith('mention') or es[1].startswith('canonical_mention')):
                event_type = event2type[es[0]]
                if event_type in event_types: # Consider only events in the KAIROS ontology
                    event_args = event_types[event_type]['args']
                    arg_name = es[1].split('.')[-2].split('_')[-1]
                    arg_nb = event_args[arg_name]
                    if not es[0] in event2args: event2args[es[0]] = {}
                    if not arg_nb in event2args[es[0]]: event2args[es[0]][arg_nb] = []
                    event2args[es[0]][arg_nb].append(es[-3].strip())

    # Check for new links
    new_link = set()
    for event in event2args:
        args = event2args[event]
        for arg_name in ARGS:
            if not arg_name in args: continue
            arg_values = args[arg_name]
            for i in range(len(arg_values)):
                for j in range(i+1, len(arg_values)):
                    a1, a2 = arg_values[i], arg_values[j]
                    if a1 != a2 and entity2type[a1] == entity2type[a2] and arg_values.count(a1) > 1 and arg_values.count(a2) > 1:
                        a1, a2 = min(a1, a2), max(a1, a2)
                        new_link.add((a1, a2))
    #
    changed = False
    if len(new_link) > 0:
        changed = True
        graph = UndirectedGraph([entity for entity in entity2type])
        print('Number of vertices: {}'.format(graph.V))
        for (node1, node2) in new_link:
            graph.addEdge(node1, node2)
        # Get connected components (with-in doc clusters)
        print('Get connected components')
        sccs = graph.getSCCs()
        assert(len(flatten(sccs)) == graph.V)
        assert(len(sccs) < graph.V)
        sccs.sort(key=lambda x: sum([len(entity2mentions[olde]) for olde in x]), reverse=True)
        # Build olde2newe
        olde2newe, newe2olde, newe2type, newe2link = {}, {}, {}, {}
        for ix, scc in enumerate(sccs):
            nb_str = str(ix)
            while len(nb_str) < 7: nb_str = '0' + nb_str
            newe = ':Entity_EDL_ENG_{}'.format(nb_str)
            for olde in scc:
                olde2newe[olde] = newe
                if not newe in newe2olde: newe2olde[newe] = []
                newe2olde[newe].append(olde)
                newe2type[newe] = entity2type[olde]
                newe2link[newe] = entity2link[olde]

        # Output new file
        with open(new_input_entity, 'w+', encoding='utf-8') as f:
            for ix in range(len(sccs)):
                nb_str = str(ix)
                while len(nb_str) < 7: nb_str = '0' + nb_str
                newe = ':Entity_EDL_ENG_{}'.format(nb_str)
                # Type line
                type_line = '\t'.join([newe, 'type', newe2type[newe]])
                f.write('{}\n'.format(type_line))
                # Other lines
                for olde in newe2olde[newe]:
                    for m in entity2mentions[olde]:
                        for line in mid2line[m]:
                            es = line.split('\t')
                            es[0] = newe
                            line = '\t'.join(es)
                            f.write('{}'.format(line))
                # link line
                link_line = '\t'.join([newe, 'link', newe2link[newe]])
                f.write('{}\n'.format(link_line))
    return changed
