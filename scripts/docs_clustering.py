import os
import time
import torch
import random

from os.path import dirname
from transformers import *
from algorithms import UndirectedGraph
from data import load_event_centric_dataset

def docs_clustering(new_input_entity, doc_ids):
    print('Document clustering')
    entity2type, entity2docs, entity2mentions = {}, {}, {}
    with open(new_input_entity, 'r') as f:
        for line in f:
            es = line.strip().split('\t')
            if es[1] == 'type':
                entity2type[es[0]] = es[-1]
            elif es[1].endswith('mention'):
                # Update entity2docs
                if not es[0] in entity2docs: entity2docs[es[0]] = set()
                doc_id = es[-2].split(':')[0]
                entity2docs[es[0]].add(doc_id)
                # Update entity2mentions
                if es[1] == 'canonical_mention':
                    if not es[0] in entity2mentions: entity2mentions[es[0]] = set()
                    entity2mentions[es[0]].add(es[-3].strip()[1:-1])

    # Filter out common entity
    deleted = set()
    for entity in entity2type:
        if entity2type[entity] == 'GPE':
            mentions = entity2mentions.get(entity, [])
            lower_mentions = [m.lower().strip() for m in mentions]

            # United States
            if 'american' in lower_mentions or 'united states' in lower_mentions or \
               'u.s.' in lower_mentions or 'u.s' in lower_mentions or 'america' in lower_mentions:
                deleted.add(entity)
    for entity in deleted:
        del(entity2type[entity])
        del(entity2docs[entity])
        del(entity2mentions[entity])

    # Find connected components
    graph = UndirectedGraph(doc_ids)
    for entity in entity2docs:
        cur_docs = list(entity2docs[entity])
        for i in range(len(cur_docs)):
            for j in range(i+1, len(cur_docs)):
                if not cur_docs[i] in doc_ids: continue
                if not cur_docs[j] in doc_ids: continue
                graph.addEdge(cur_docs[i], cur_docs[j])
    sccs = graph.getSCCs()

    # clusters
    clusters = [list(scc) for scc in sccs]

    return clusters
