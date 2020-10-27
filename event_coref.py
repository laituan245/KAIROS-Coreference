import os
import time
import torch
import copy
import random

from constants import *
from os.path import dirname
from utils import load_tokenizer_and_model, get_predicted_antecedents, flatten, create_dir_if_not_exist, read_event_types
from data import EventCentricDocument, EventCentricDocumentPair, load_event_centric_dataset
from algorithms import UndirectedGraph

INTERMEDIATE_PRED_EVENT_PAIRS = 'event_pred_pairs.txt'

def args_overlap(argi, argj):
    if argi is None: return False
    if argj is None: return False
    return len(argi.intersection(argj)) > 0

# Main Function
def event_coref(cs_path, json_dir, output_path, original_input_entity, new_input_entity, filtered_doc_ids, clusters):
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
                    event2type[es[0]] = es[-2]
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
        f = open(INTERMEDIATE_PRED_EVENT_PAIRS, 'w+')
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

    # Build mid2type
    mid2type = {}
    for i in range(len(mentions)):
        mid2type[mentions[i]['mention_id']] = mentions[i]['event_type']

    # Add edges from INTERMEDIATE_PRED_EVENT_PAIRS (all edges will be in-doc)
    edge_pairs = set()
    print('Add edges from INTERMEDIATE_PRED_EVENT_PAIRS')
    with open(INTERMEDIATE_PRED_EVENT_PAIRS, 'r') as f:
        for line in f:
            es = line.split('\t')
            node1, node2 = es[0].strip(), es[1].strip()
            node1_args = event2args.get(node1, {})
            node2_args = event2args.get(node2, {})
            # Rules
            if mid2type[node1] == 'Justice.Sentence.Unspecified' and mid2type[node2] == 'Justice.ReleaseParole.Unspecified': continue
            if mid2type[node1] == 'Justice.ReleaseParole.Unspecified' and mid2type[node2] == 'Justice.Sentence.Unspecified': continue

            graph.addEdge(node1, node2)
            edge_pairs.add((node1, node2))
            edge_pairs.add((node2, node1))

    # Add edge between two event mentions if they have the same subtype and same args1/args2
    for i in range(len(mentions)):
        for j in range(len(mentions)):
            if i == j: continue
            if originaldoc2cluster[mentions[i]['doc_id']] != originaldoc2cluster[mentions[j]['doc_id']]: continue
            subtypei, subtypej = mentions[i]['event_type'], mentions[j]['event_type']
            typei, typej = subtypei[:subtypei.rfind('.')], subtypej[:subtypej.rfind('.')]
            if typei != typej: continue
            type = typei = typej
            # Extract arguments of two mentions
            args_seti = event2args.get(mentions[i]['mention_id'], {})
            args_setj = event2args.get(mentions[j]['mention_id'], {})
            # Consider special event types (e.g., attack, justice, die ...)
            cond_met = False
            # Assuming all crime investigations are about 1 central crime / attack event
            if type in ['Justice.InvestigateCrime']:
                cond_met = True
            # considering <arg1>
            if type in ['Life.Die']:
                if args_overlap(args_seti.get('<arg1>'), args_setj.get('<arg1>')):
                    cond_met = True
            # considering <arg2>
            if type in ['Conflict.Attack', 'Movement.Transportation', 'Justice.Sentence', 'Justice.TrialHearing']:
                if args_overlap(args_seti.get('<arg2>'), args_setj.get('<arg2>')):
                    cond_met = True
            # considering Attack.DetonateExplode with Attack.Unspecified
            if 'Attack.DetonateExplode' in subtypei and 'Attack.Unspecified' in subtypej:
                if args_overlap(args_seti.get('<arg3>'), args_setj.get('<arg3>')):
                    cond_met = True
                if args_overlap(args_seti.get('<arg4>'), args_setj.get('<arg3>')):
                    cond_met = True
            # considering Attack.DetonateExplode
            if 'Attack.DetonateExplode' in subtypei and 'Attack.DetonateExplode' in subtypej:
                for ix in range(5):
                    argi = args_seti.get('<arg{}>'.format(i+1))
                    argj = args_setj.get('<arg{}>'.format(i+1))
                    if args_overlap(argi, argj): cond_met = True
            if cond_met:
                mid_i, mid_j = mentions[i]['mention_id'], mentions[j]['mention_id']
                edge_pairs.add((mid_i, mid_j))
                edge_pairs.add((mid_j, mid_i))
                graph.addEdge(mid_i, mid_j)

    # Get connected components (with-in doc clusters)
    print('Get connected components')
    clusters = graph.getSCCs()
    assert(len(flatten(clusters)) == graph.V)

    # If there is one single big "explode" cluster and several singleton "explode" clusters
    # then merge them all together
    doc_clusters = set(originaldoc2cluster.values())
    for doc_cluster in doc_clusters:
        big_explodes, singleton_explodes = [], []
        for index, c in enumerate(clusters):
            did = list(c)[0].split(':')[0]
            if not originaldoc2cluster[did] == doc_cluster: continue
            types = [mid2type[mid] for mid in c]
            if types.count('Conflict.Attack.DetonateExplode') > 1: big_explodes.append(index)
            if types.count('Conflict.Attack.DetonateExplode') == 1 and len(types) == 1: singleton_explodes.append(index)
        if len(big_explodes) == 1:
            big_explode_index = big_explodes[0]
            for single_index in singleton_explodes:
                clusters[big_explode_index] = clusters[big_explode_index].union(copy.deepcopy(clusters[single_index]))
                clusters[single_index] = []
            clusters = [c for c in clusters if len(c) > 0]

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
