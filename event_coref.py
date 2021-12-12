import os
import time
import torch
import copy
import random

from constants import *
from os.path import dirname, join
from utils import load_tokenizer_and_model, get_predicted_antecedents, flatten, create_dir_if_not_exist, read_event_types
from data import EventCentricDocument, EventCentricDocumentPair, load_event_centric_dataset
from algorithms import UndirectedGraph

BOMBING_KEYWORDS = ['gone off', 'bomb', 'detonate']

def args_overlap(argi, argj):
    if argi is None: return False
    if argj is None: return False
    return len(argi.intersection(argj)) > 0

def locstr_to_loc(loc_str):
    doc_id, offset_info = loc_str.split(':')
    start, end = offset_info.split('-')
    start, end = int(start), int(end)
    return (doc_id, start, end)

# Main Function
def event_coref(cs_path, json_dir, output_path, original_input_entity, new_input_entity, filtered_doc_ids, clusters, english_docs, spanish_docs):
    create_dir_if_not_exist(dirname(output_path))

    # Build olde2mid
    olde2mid = {}
    with open(original_input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            es = line.split('\t')
            if es[1].startswith('mention') or es[1].startswith('canonical_mention'):
                olde2mid[es[0]] = es[-2]

    # Build mid2eid
    mid2eid = {}
    with open(new_input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            es = line.split('\t')
            if es[1].endswith('mention'):
                mid2eid[es[-2]] = es[0]

    # Read info of event_types
    event_types = read_event_types('resources/event_types.tsv')

    # Read old event_cs file
    event2type, event2args, oldevs2mid, event2text = {}, {}, {}, {}
    with open(cs_path, 'r', encoding='utf-8') as f:
        for line in f:
            es = line.strip().split('\t')
            if len(es) <= 4:
                if es[1] == 'type':
                    ev_type = es[2]
                    if '#' in ev_type:
                        ev_type = ev_type[ev_type.rfind('#')+1:]
                    event2type[es[0]] = ev_type
                continue
            if not (es[1].startswith('mention') or es[1].startswith('canonical_mention')):
                event_type = event2type[es[0]]
                if event_type in event_types: # Consider only events in the KAIROS ontology
                    event_args = event_types[event_type]['args']
                    arg_name = es[1].split('.')[-2].split('_')[-1]
                    arg_nb = event_args[arg_name]
                    mid = oldevs2mid[es[0]]
                    if not mid in event2args: event2args[mid] = {}
                    if not arg_nb in event2args[mid]: event2args[mid][arg_nb] = set()
                    event2args[mid][arg_nb].add(mid2eid[olde2mid[es[2]]])
            else:
                oldevs2mid[es[0]] = es[-2].strip()

    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(EVENT_MODEL)

    # Load dataset
    print('Loading dataset')
    docs = load_event_centric_dataset(tokenizer, cs_path, json_dir, filtered_doc_ids)
    _clusters = []
    for i in range(len(clusters)): _clusters.append([])
    for doc in docs:
        doc_id = doc.doc_id
        for ix in range(len(clusters)):
            found = False
            for j in range(len(clusters[ix])):
                if clusters[ix][j] in doc_id:
                    found = True
                    _clusters[ix].append(doc_id)
                    break
            if found: break
    clusters = _clusters

    # Build mentions and id2mention
    mentions, id2mention = [], {}
    for doc in docs:
        mentions += doc.event_mentions
    for m in mentions:
        id2mention[m['mention_id']] = m

    # Build doc2cluster
    doc2cluster, originaldoc2cluster = {}, {}
    for ix, c in enumerate(clusters):
        for doc_id in c:
            doc2cluster[doc_id] = ix
            originaldoc2cluster[doc_id[:doc_id.rfind('_part')]] = ix

    # Apply the coref model
    doc_pairs_ctx = 0
    start_time = time.time()
    if True:
        predicted_pairs = set()
        # Main loop
        for i in range(len(docs)):
            doci = docs[i]
            end_range = len(docs) if len(clusters[doc2cluster[doci.doc_id]]) > 1 else len(docs)+1
            for j in range(i+1, end_range):
                if j == len(docs):
                    # Dummy doc
                    docj = EventCentricDocument(doci.doc_id, [], [])
                else:
                    docj = docs[j]
                if len(doci.words) == 0 and len(docj.words) == 0: continue
                if doc2cluster[doci.doc_id] != doc2cluster[docj.doc_id]: continue
                inst = EventCentricDocumentPair(doci, docj, tokenizer)
                if len(inst.event_mentions) == 0: continue
                preds = model(inst, is_training=False)[1]
                preds = [x.cpu().data.numpy() for x in preds]
                mention_starts, mention_ends, top_antecedents, top_antecedent_scores = preds
                predicted_antecedents, predicted_scores = \
                    get_predicted_antecedents(top_antecedents, top_antecedent_scores, True)

                # Decide cluster from predicted_antecedents
                doc_entities = inst.event_mentions
                for ix, (s, e) in enumerate(zip(mention_starts, mention_ends)):
                    if predicted_antecedents[ix] >= 0:
                        antecedent_idx = predicted_antecedents[ix]
                        mention_1 = doc_entities[ix]
                        mention_2 = doc_entities[antecedent_idx]
                        m1_id = mention_1['mention_id']
                        m2_id = mention_2['mention_id']
                        # Add to predicted_pairs
                        predicted_pairs.add((m1_id, m2_id))

                # Update doc_pairs_ctx
                doc_pairs_ctx += 1
                if doc_pairs_ctx % 1000 == 0:
                    print('doc_pairs_ctx = {}'.format(doc_pairs_ctx))
                    print("--- Ran for %s seconds ---" % (time.time() - start_time))
        f.close()
    print("--- Applying the event coref model took %s seconds ---" % (time.time() - start_time))

    # Build graph
    graph = UndirectedGraph([m['mention_id'] for m in mentions])
    print('Number of vertices: {}'.format(graph.V))

    # Build mid2type
    mid2type = {}
    for i in range(len(mentions)):
        mid2type[mentions[i]['mention_id']] = mentions[i]['event_type']

    # Add edges
    edge_pairs = set()
    print('Add edges')
    for (node1, node2) in predicted_pairs:
        # Arguments
        node1_args = event2args.get(node1, {})
        node2_args = event2args.get(node2, {})
        # General types
        node1_type = mid2type[node1]
        node1_type = node1_type[:node1_type.rfind('.')]
        node2_type = mid2type[node2]
        node2_type = node2_type[:node2_type.rfind('.')]
        # Rule 1: Different general types -> Skip
        if node1_type != node2_type:
            continue
        # Rule 2: Incompatible <arg1> and <arg2> -> Skip
        should_skip = False
        for arg_nb in [1, 2]:
            arg_name = '<arg{}>'.format(arg_nb)
            if arg_name in node1_args and arg_name in node2_args:
                arg_vals_1 = node1_args[arg_name]
                arg_vals_2 = node2_args[arg_name]
                if not args_overlap(arg_vals_1, arg_vals_2):
                    should_skip = True
                    break
        if should_skip:
            continue
        # Append edge
        graph.addEdge(node1, node2)
        edge_pairs.add((node1, node2))
        edge_pairs.add((node2, node1))

    # Get connected components (with-in doc clusters)
    print('Get connected components')
    clusters = graph.getSCCs()
    assert(len(flatten(clusters)) == graph.V)

    # Read the original event.cs to get lines ...
    mid2lines, oid2mid = {}, {}
    with open(cs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            es = line.split('\t')
            if len(line) == 0: continue
            if len(es) <= 4: continue
            if es[1].startswith('mention') or es[1].startswith('canonical_mention'):
                mid = es[-2]
                if not mid in mid2lines: mid2lines[mid] = []
                mid2lines[mid].append(es)
                oid2mid[es[0].strip()] = mid
            else:
                mid = oid2mid[es[0].strip()]
                mid2lines[mid].append(es)

    # Outputs
    prefix = '::Event_'
    clusters.sort(key=lambda x: len(x), reverse=True)
    with open(output_path, 'w+', encoding='utf-8') as f:
        for ix, cluster in enumerate(clusters):
            ix_str = str(ix)
            while len(ix_str) < 7: ix_str = '0' + ix_str
            es_0 = prefix + ix_str
            # Element in clusters
            if len(cluster) == 1:
                doc_id = locstr_to_loc(id2mention[list(cluster)[0]]['mention_id'])[0]
                if doc_id in spanish_docs:
                    continue

            for m in cluster:
                mention = id2mention[m]
                mid = mention['mention_id']
                assert(len(mid2lines[mid]) > 0)
                # Type line
                line_es = [es_0, 'type', mention['event_type']]
                type_line = '\t'.join(line_es)
                f.write('{}\n'.format(type_line))
                # Other lines
                for line in mid2lines[mid]:
                    line[0] = es_0
                    mention_line = '\t'.join(line)
                    f.write('{}\n'.format(mention_line))

    model.to(torch.device('cpu'))

    # Delete unused objects
    del(model)
    del(tokenizer)
