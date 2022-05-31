import json
import argparse
import time

from algorithms import UndirectedGraph
from utils import *
from data.helpers import divide_event_docs
from data import EventCentricDocument, EventCentricDocumentPair, load_event_centric_dataset
from constants import EVENT_MODEL
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fp', default='resources/30may/0206_args.json')
    parser.add_argument('--output_fp', default='clusters.txt')
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(EVENT_MODEL)
    print('Loaded tokenizer and model')

    # Data preparation
    all_words, all_mentions, sent_lens = read_sent_level_event_extraction_input(args.input_fp)
    for m in all_mentions:
        m['mention_id'] = '{}:{}-{}'.format(m['sent_id'], m['start_char'], m['end_char'])
    raw_docs = divide_event_docs(all_words, all_mentions, sent_lens)
    # Update test_docs
    test_docs = []
    for ix, (cur_words, cur_mentions) in enumerate(raw_docs):
        test_doc = EventCentricDocument('part_{}'.format(ix), cur_words, cur_mentions)
        test_doc.tokenize(tokenizer)
        test_docs.append(test_doc)
    test_docs.sort(key=lambda x: x.doc_id)
    docs = test_docs

    # Info
    doc_ids = [x.doc_id for x in test_docs]
    print('Number of splitted docs: {}'.format(len(doc_ids)))

    # Apply the coref model 
    doc_pairs_ctx = 0
    start_time = time.time()
    if True:
        predicted_pairs = set()
        # Main loop
        for i in range(len(docs)):
            doci = docs[i]
            end_range = 2 if len(docs) == 1 else len(docs)
            for j in range(i+1, end_range):
                if j == len(docs):
                    # Dummy doc
                    docj = EventCentricDocument(doci.doc_id, [], [])
                else:
                    docj = docs[j]
                if len(doci.words) == 0 and len(docj.words) == 0: continue
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
    print("--- Applying the event coref model took %s seconds ---" % (time.time() - start_time))


    # Build graph
    mentions = all_mentions
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
        # General types
        node1_type = mid2type[node1]
        node1_type = node1_type[:node1_type.rfind('.')]
        node2_type = mid2type[node2]
        node2_type = node2_type[:node2_type.rfind('.')]
        # Rule 1: Different general types -> Skip
        if node1_type != node2_type:
            continue
        # Append edge
        graph.addEdge(node1, node2)
        edge_pairs.add((node1, node2))
        edge_pairs.add((node2, node1))

    # Get connected components (with-in doc clusters)
    print('Get connected components')
    clusters = graph.getSCCs()
    assert(len(flatten(clusters)) == graph.V)

    print('Number of clusters: {}'.format(len(clusters)))

    # Output
    with open(args.output_fp, 'w+') as f:
        for cluster in clusters:
            f.write('{}\n'.format(json.dumps(list(cluster))))
