import os
import time
import torch
import random

from constants import *
from os.path import dirname
from utils import load_tokenizer_and_model, get_predicted_antecedents, flatten, create_dir_if_not_exist
from data import EventCentricDocumentPair, load_event_centric_dataset
from algorithms import UndirectedGraph

INTERMEDIATE_PRED_EVENT_PAIRS = 'event_pred_pairs.txt'

# Helper Function
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

# Main Function
def event_coref(cs_path, json_dir, output_path, language, original_input_entity, new_input_entity, filtered_doc_ids):
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
    event2type, event2args, oldevs2mid = {}, {}, {}
    with open(cs_path, 'r', encoding='utf-8') as f:
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
                    arg_nb = event_args[arg_name]
                    mid = oldevs2mid[es[0]]
                    if not mid in event2args: event2args[mid] = {}
                    if not arg_nb in event2args[mid]: event2args[mid][arg_nb] = []
                    event2args[mid][arg_nb].append(mid2eid[olde2mid[es[2]]])
            else:
                oldevs2mid[es[0]] = es[-2].strip()

    # Load tokenizer and model
    if language == 'en': tokenizer, model = load_tokenizer_and_model(EN_EVENT_MODEL)
    elif language == 'es': tokenizer, model = load_tokenizer_and_model(ES_EVENT_MODEL)

    # Load dataset
    print('Loading dataset')
    docs, clusters = load_event_centric_dataset(tokenizer, cs_path, json_dir, filtered_doc_ids)

    # Build mentions and id2mention
    mentions, id2mention = [], {}
    for doc in docs:
        mentions += doc.event_mentions
    for m in mentions:
        id2mention[m['mention_id']] = m

    # Build doc2cluster
    doc2cluster = {}
    for ix, c in enumerate(clusters):
        for doc_id in c:
            doc2cluster[doc_id] = ix

    # Apply the coref model
    doc_pairs_ctx = 0
    start_time = time.time()
    if True:
        f = open(INTERMEDIATE_PRED_EVENT_PAIRS, 'w+')
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                doci, docj = docs[i], docs[j]
                if doc2cluster[doci.doc_id] != doc2cluster[docj.doc_id]: continue
                inst = EventCentricDocumentPair(doci, docj, tokenizer)
                if len(inst.event_mentions) == 0: continue
                preds = model(inst, is_training=False)[1]
                preds = [x.cpu().data.numpy() for x in preds]
                mention_starts, mention_ends, top_antecedents, top_antecedent_scores = preds
                predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)

                # Decide cluster from predicted_antecedents
                doc_entities = inst.event_mentions
                predicted_clusters, m2cluster = [], {}
                for ix, (s, e) in enumerate(zip(mention_starts, mention_ends)):
                    if predicted_antecedents[ix] < 0:
                        cluster_id = len(predicted_clusters)
                        predicted_clusters.append([doc_entities[ix]])
                    else:
                        antecedent_idx = predicted_antecedents[ix]
                        p_s, p_e = mention_starts[antecedent_idx], mention_ends[antecedent_idx]
                        cluster_id = m2cluster[(p_s, p_e)]
                        predicted_clusters[cluster_id].append(doc_entities[ix])
                    m2cluster[(s,e)] = cluster_id

                # Extract predicted pairs
                for c in predicted_clusters:
                    if len(c) <= 1: continue
                    for ix in range(len(c)):
                        for jx in range(ix+1, len(c)):
                            f.write('{}\t{}\n'.format(c[ix]['mention_id'], c[jx]['mention_id']))

                # Update doc_pairs_ctx
                doc_pairs_ctx += 1
                if doc_pairs_ctx % 1000 == 0:
                    print('doc_pairs_ctx = {}'.format(doc_pairs_ctx))
                    print("--- Ran for %s seconds ---" % (time.time() - start_time))
        f.close()
    print("--- Applying the event coref model took %s seconds ---" % (time.time() - start_time))

    # Build clusters from INTERMEDIATE_PRED_EVENT_PAIRS
    graph = UndirectedGraph([m['mention_id'] for m in mentions])
    print('Number of vertices: {}'.format(graph.V))

    # Add edges from INTERMEDIATE_PRED_EVENT_PAIRS (all edges will be in-doc)
    print('Add edges from INTERMEDIATE_PRED_EVENT_PAIRS')
    with open(INTERMEDIATE_PRED_EVENT_PAIRS, 'r') as f:
        for line in f:
            es = line.split('\t')
            node1, node2 = es[0].strip(), es[1].strip()
            node1_args = event2args.get(node1, {})
            node2_args = event2args.get(node2, {})
            # Filtering by <arg1> (if both have <arg1>)
            if '<arg1>' in node1_args and '<arg1>' in node2_args:
                compatible = False
                for a1 in node1_args['<arg1>']:
                    for a2 in node2_args['<arg1>']:
                        if a1 == a2: compatible = True
                if not compatible: continue
            # Filtering by <arg2> (if both have <arg2>)
            if '<arg2>' in node1_args and '<arg2>' in node2_args:
                compatible = False
                for a1 in node1_args['<arg2>']:
                    for a2 in node2_args['<arg2>']:
                        if a1 == a2: compatible = True
                if not compatible: continue
            graph.addEdge(node1, node2)

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

    # Remove INTERMEDIATE_PRED_EVENT_PAIRS
    os.remove(INTERMEDIATE_PRED_EVENT_PAIRS)
