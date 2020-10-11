import os
import time
import torch
import random

from constants import *
from argparse import ArgumentParser
from utils import load_tokenizer_and_model, get_predicted_antecedents, flatten
from data import EventCentricDocumentPair, load_event_centric_dataset
from algorithms import UndirectedGraph

INTERMEDIATE_PRED_PAIRS = 'event_pred_pairs.txt'

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--cs_path', default='/shared/nas/data/m1/yinglin8/kairos/result/all_oct_4/m1_m2/cs/event.cs')
    parser.add_argument('-j', '--json_dir', default='/shared/nas/data/m1/yinglin8/kairos/result/all_oct_4/m1_m2/json')
    parser.add_argument('-o', '--output_path', default='/shared/nas/data/m1/tuanml2/kairos/output/corrected_format/event.cs')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # DEBUG Mode
    if args.debug:
        args.cs_path = 'resources/samples/event.cs'
        args.json_dir = 'resources/samples/json'
        args.output_path = 'event.cs'

    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(EVENT_MODEL)

    # Load dataset
    print('Loading dataset')
    docs, clusters = load_event_centric_dataset(tokenizer, args.cs_path, args.json_dir)

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
    if not os.path.exists(INTERMEDIATE_PRED_PAIRS):
        f = open(INTERMEDIATE_PRED_PAIRS, 'w+')
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

    # Build clusters from INTERMEDIATE_PRED_PAIRS
    graph = UndirectedGraph([m['mention_id'] for m in mentions])
    print('Number of vertices: {}'.format(graph.V))

    # Add edges from INTERMEDIATE_PRED_PAIRS (all edges will be in-doc)
    print('Add edges from INTERMEDIATE_PRED_PAIRS')
    with open(INTERMEDIATE_PRED_PAIRS, 'r') as f:
        for line in f:
            es = line.split('\t')
            node1, node2 = es[0].strip(), es[1].strip()
            graph.addEdge(node1, node2)

    # Get connected components (with-in doc clusters)
    print('Get connected components')
    clusters = graph.getSCCs()
    assert(len(flatten(clusters)) == graph.V)

    # Read the original event.cs to get lines ...
    mid2lines, oid2mid = {}, {}
    with open(args.cs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            es = line.split('\t')
            if len(line) == 0: continue
            if len(es) <= 3: continue
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
    with open(args.output_path, 'w+', encoding='utf-8') as f:
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
