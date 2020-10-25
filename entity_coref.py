import os
import time
import torch

from constants import *
from argparse import ArgumentParser
from utils import load_tokenizer_and_model, get_predicted_antecedents, flatten, create_dir_if_not_exist
from data import load_entity_centric_dataset
from algorithms import UndirectedGraph
from os.path import join, dirname

INTERMEDIATE_PRED_ENTITY_PAIRS = 'entity_pred_pairs.txt'

# Helper Function
def get_cluster_labels(clusters, id2mention, field):
    clusterlabels, nil_ctx = [], 0
    for c in clusters:
        count = {}
        for mid in c:
            m = id2mention[mid]
            if field == 'fb_id' and field in m and m[field].startswith('NIL'): continue
            if field in m: count[m[field]] = count.get(m[field], 0) + 1
        if len(count) == 0:
            label = 'no_label_{}'.format(str(nil_ctx))
            nil_ctx += 1
        else:
            label = max(count, key=lambda k: count[k])
        clusterlabels.append(label)
    return clusterlabels

def entity_coref(cs_path, json_dir, fb_linking_path, output_path, language, filtered_doc_ids, clusters=None):
    create_dir_if_not_exist(dirname(output_path))

    # Read the original entity.cs
    mid2lines, oe2mid, cur_type = {}, {}, None
    with open(cs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            es = line.split('\t')
            if len(line) == 0: continue
            if len(es) <= 4:
                if es[1].strip().lower() == 'type':
                    cur_type = es[2].strip()
                continue
            mid = es[-2]
            oe2mid[es[0]] = mid
            if not mid in mid2lines: mid2lines[mid] = []
            if not es[1].startswith('canonical_mention'):
                mid2lines[mid].append(es)

    # Read the original relation.cs file (in the same directory as entity.cs)
    relation_cs = join(dirname(cs_path), 'relation.cs')
    relation_pairs = set()
    with open(relation_cs, 'r', encoding='utf-8') as f:
        for line in f:
            es = line.strip().split('\t')
            arg0, arg1 = oe2mid[es[0]], oe2mid[es[2]]
            relation_pairs.add((arg0, arg1))
            relation_pairs.add((arg1, arg0))

    # Load tokenizer and model
    if language == 'en': tokenizer, model = load_tokenizer_and_model(EN_ENTITY_MODEL)
    elif language == 'es': tokenizer, model = load_tokenizer_and_model(ES_ENTITY_MODEL)

    # Load dataset
    print('Loading dataset')
    entities, dataset = load_entity_centric_dataset(tokenizer, cs_path, json_dir, fb_linking_path, filtered_doc_ids)
    mentions = flatten([e['mentions'].values() for e in entities.values()])

    # Build mid2type
    mid2type = {}
    for e in entities.values():
        e_type = e['type']
        for m in e['mentions']:
            mid2type[m] = e_type

    # Build id2mention
    id2mention, fb2mentions = {}, {}
    for m in mentions:
        id2mention[m['mention_id']] = m
        if 'fb_id' in m:
            if not m['fb_id'] in fb2mentions: fb2mentions[m['fb_id']] = []
            fb2mentions[m['fb_id']].append(m)

    # Apply the coref model
    start_time = time.time()
    if True:
        f = open(INTERMEDIATE_PRED_ENTITY_PAIRS, 'w+')
        with torch.no_grad():
            for doc_index, tensorized_example in enumerate(dataset.tensorized_examples[TEST]):
                print('doc_index = {}'.format(doc_index))
                doc_entities = dataset.data[doc_index].entity_mentions
                preds = model(*tensorized_example)[1]
                preds = [x.cpu().data.numpy() for x in preds]
                mention_starts, mention_ends, top_antecedents, top_antecedent_scores = preds
                predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)

                # Decide cluster from predicted_antecedents
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
                    for i in range(len(c)):
                        for j in range(i+1, len(c)):
                            f.write('{}\t{}\n'.format(c[i]['mention_id'], c[j]['mention_id']))
        f.close()
    print("--- Applying the entity coref model took %s seconds ---" % (time.time() - start_time))

    # Build clusters from INTERMEDIATE_PRED_ENTITY_PAIRS
    graph = UndirectedGraph([m['mention_id'] for m in mentions])
    print('Number of vertices: {}'.format(graph.V))

    # Add edges from INTERMEDIATE_PRED_ENTITY_PAIRS (all edges will be in-doc)
    print('Add edges from INTERMEDIATE_PRED_ENTITY_PAIRS')
    with open(INTERMEDIATE_PRED_ENTITY_PAIRS, 'r') as f:
        for line in f:
            es = line.split('\t')
            node1, node2 = es[0].strip(), es[1].strip()
            if (node1, node2) in relation_pairs or (node2, node1) in relation_pairs: continue
            if mid2type[node1] != mid2type[node2]: continue
            # Add edges
            graph.addEdge(node1, node2)
    # Get connected components (with-in doc clusters)
    print('Get connected components')
    sccs = graph.getSCCs()
    assert(len(flatten(sccs)) == graph.V)

    # EM loop
    clusters = sccs
    while True:
        print('EM loop | Nb clusters = {}'.format(len(clusters)))
        cluster_labels = get_cluster_labels(clusters, id2mention, field='fb_id')
        # Build label2clusters
        label2clusters = {}
        for label, c in zip(cluster_labels, clusters):
            if not label in label2clusters: label2clusters[label] = []
            label2clusters[label].append(c)
        # Build new clusters
        new_clusters = []
        for l in set(cluster_labels):
            new_clusters.append(flatten(label2clusters[l]))
        # Stopping condition
        if len(new_clusters) == len(clusters):
            break
        clusters = new_clusters

    # Read the fb_linking_path to get canonical_mention and mention
    in_fb_linking = set()
    with open(fb_linking_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            es = line.split('\t')
            if len(line) == 0: continue
            if len(es) <= 3: continue
            mid = es[-2]
            if not mid in in_fb_linking:
                in_fb_linking.add(mid)
                mid2lines[mid] = []
            mid2lines[mid].append(es)

    # Outputs
    nil_count = 0
    prefix = ':Entity_EDL_ENG_'
    clusters.sort(key=lambda x: len(x), reverse=True)
    c_types = get_cluster_labels(clusters, id2mention, field='type')
    c_links = get_cluster_labels(clusters, id2mention, field='fb_id')
    with open(output_path, 'w+', encoding='utf-8') as f:
        for ix, (type, link, cluster) in enumerate(zip(c_types, c_links, clusters)):
            ix_str = str(ix)
            while len(ix_str) < 7: ix_str = '0' + ix_str
            es_0 = prefix + ix_str
            # Type
            type_line = '\t'.join([es_0, 'type', type])
            f.write('{}\n'.format(type_line))
            # Element in clusters
            for m in cluster:
                for line in mid2lines[m]:
                    line[0] = es_0
                    mention_line = '\t'.join(line)
                    f.write('{}\n'.format(mention_line))
            # Link
            if not link.startswith('no_label_'):
                link_line = '\t'.join([es_0, 'link', link])
                f.write('{}\n'.format(link_line))
            else:
                nil_count += 1
                link = str(nil_count)
                while len(link) < 9: link = '0' + link
                link = 'NIL' + link
                link_line = '\t'.join([es_0, 'link', link])
                f.write('{}\n'.format(link_line))

    # Remove INTERMEDIATE_PRED_ENTITY_PAIRS
    os.remove(INTERMEDIATE_PRED_ENTITY_PAIRS)
