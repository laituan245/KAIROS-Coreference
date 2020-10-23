import os
import time
import torch
import random

from os.path import dirname
from transformers import *
from algorithms import UndirectedGraph
from data import load_event_centric_dataset, read_cs, locstr_to_loc

def docs_clustering(linking_output, doc_ids):
    print('Document clustering')

    filtered_entities = []
    e2info = read_cs(linking_output, skip_firstline=True)
    for e in e2info:
        entity = e2info[e]

        # Skip NIL.... entities
        if entity['link'].startswith('NIL'): continue

        # Skip US / United States since it is too common
        if entity['type'] == 'GPE':
            m_texts = [m['text'].lower() for m in entity['mentions'].values()]
            if 'american' in m_texts or 'united states' in m_texts or 'u.s.' in m_texts or 'u.s' in m_texts or 'america' in m_texts:
                # English
                continue
            if 'estados unidos' in m_texts or 'eeuu' in m_texts:
                # Spanish
                continue

        # Update filtered_entities
        filtered_entities.append(entity)

    # Find connected components
    freq_ctx = {}
    graph = UndirectedGraph(doc_ids)
    for entity in filtered_entities:
        cur_docs = set([m['doc_id'] for m in entity['mentions'].values()])
        cur_docs = list(cur_docs)
        cur_docs = [d for d in cur_docs if d in doc_ids]
        if len(cur_docs) <= 1: continue
        for i in range(len(cur_docs)):
            for j in range(i+1, len(cur_docs)):
                if not cur_docs[i] in doc_ids: continue
                if not cur_docs[j] in doc_ids: continue
                cur_docs[i], cur_docs[j] = min(cur_docs[i], cur_docs[j]), max(cur_docs[i], cur_docs[j])
                pair = (cur_docs[i], cur_docs[j])
                freq_ctx[pair] = freq_ctx.get(pair, 0) + 1
                if freq_ctx[pair] > 1:
                    graph.addEdge(cur_docs[i], cur_docs[j])
    sccs = graph.getSCCs()

    # clusters
    clusters = [list(scc) for scc in sccs]
    print('clusters are : {}'.format(clusters))
    return clusters
