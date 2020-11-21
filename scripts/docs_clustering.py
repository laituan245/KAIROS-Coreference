import json
import numpy as np
from os import listdir
from sklearn.cluster import DBSCAN
from os.path import isfile, join
from sentence_transformers import SentenceTransformer, util

def locstr_to_loc(loc_str):
    doc_id, offset_info = loc_str.split(':')
    start, end = offset_info.split('-')
    start, end = int(start), int(end)
    return (doc_id, start, end)

def read_json_docs(base_path):
    doc2sents = {}
    for f in listdir(base_path):
        if isfile(join(base_path, f)) and f.endswith('json'):
            file_path = join(base_path, f)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    doc_id = data['doc_id']
                    if not doc_id in doc2sents: doc2sents[doc_id] = []
                    sents, tokens, token_ids = [], data['tokens'], data['token_ids']
                    for token, token_id in zip(tokens, token_ids):
                        _doc_id, start, end = locstr_to_loc(token_id)
                        assert(_doc_id == doc_id)
                        sents.append((token, start, end))
                    doc2sents[doc_id].append(sents)
    print('Number of docs from {} is {}'.format(base_path, len(doc2sents)))
    return doc2sents


def flatten(l):
    return [item for sublist in l for item in sublist]

def docs_clustering(json_dir, distracted_docs=set(), verbose=True):
    json_base_path = json_dir
    # Build doc2text and doc2sents
    doc2text = {}
    doc2sents = read_json_docs(json_base_path)
    for doc in doc2sents:
        words = flatten(doc2sents[doc])
        doc2text[doc] = ' '.join([w[0] for w in words])

    # model
    model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

    # doc_ids, texts, embeddings
    skipped = 0
    doc_ids, texts, embeddings = [], [], []
    for doc in doc2text:
        if doc in distracted_docs:
            skipped += 1
            continue
        doc_ids.append(doc)
        texts.append(doc2text[doc])
        embeddings.append(model.encode(doc2text[doc]))
    all_doc_ids = set(doc_ids)
    print('Skipped {} docs'.format(skipped))

    # Clustering
    total, ctx = 0,0
    X = np.zeros((len(doc_ids), len(doc_ids)))
    for i in range(len(doc_ids)):
        for j in range(len(doc_ids)):
            X[i,j] = max(0, 1 - util.pytorch_cos_sim(embeddings[i], embeddings[j]))
            total += X[i,j]
            ctx += 1

    # DBSCAN
    clustering = DBSCAN(eps=0.45, min_samples=2, metric='precomputed').fit(X)
    labels = clustering.labels_.tolist()

    # clusters
    clusters = []
    for ix in range(len(labels)):
        cur_label = labels[ix]
        if cur_label >= len(clusters): clusters.append([])
        clusters[cur_label].append(doc_ids[ix])

    # Sorting
    clusters.sort(key=lambda x: len(x), reverse=True)

    if verbose:
        print('There are {} clusters: {}'.format(len(clusters), clusters))

    return clusters