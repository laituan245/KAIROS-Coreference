import json
import numpy as np
from os import listdir
from sklearn.cluster import DBSCAN
from os.path import isfile, join
from sentence_transformers import SentenceTransformer, util

ATTACK_EN_KEYWORDS = ['bomb', 'bombs', 'explosive', 'explosives', 'drone', 'drones',
                      'strike', 'strikes', 'attack', 'attacks', 'shoot', 'shoots']
ATTACK_ES_KEYWORDS = ['bomba', 'bombas', 'explosivo', 'explosiva', 'explosivos',
                      'dron', 'drones', 'huelga', 'huelgas', 'ataque', 'ataques', 'disparan']


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

def docs_filtering(json_dir, language):
    json_base_path = json_dir
    # Build doc2text and doc2sents
    doc2text = {}
    doc2sents = read_json_docs(json_base_path)
    for doc in doc2sents:
        words = flatten(doc2sents[doc])
        doc2text[doc] = ' '.join([w[0] for w in words])

    # model
    assert(language in ['en', 'es'])
    if language == 'en':
        model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    if language == 'es':
        model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

    # doc_ids, texts, embeddings
    doc_ids, texts, embeddings = [], [], []
    for doc in doc2text:
        doc_ids.append(doc)
        texts.append(doc2text[doc])
        embeddings.append(model.encode(doc2text[doc]))
    all_doc_ids = set(doc_ids)

    return all_doc_ids, set()


    # Clustering
    total, ctx = 0,0
    X = np.zeros((len(doc_ids), len(doc_ids)))
    for i in range(len(doc_ids)):
        for j in range(len(doc_ids)):
            X[i,j] = max(0, 1 - util.pytorch_cos_sim(embeddings[i], embeddings[j]))
            total += X[i,j]
            ctx += 1

    # DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='precomputed').fit(X)
    labels = clustering.labels_.tolist()

    distracted_docs = []
    for l, doc_id in zip(labels, doc_ids):
        if l < 0:
            distracted_docs.append(doc_id)

    distracted_docs = set(distracted_docs)
    filtered_docs = all_doc_ids - distracted_docs

    print('[AFTER DBSCAN FILTERING] Remaining doc ids: {}'.format(filtered_docs))
    print('Distracted doc ids: {}'.format(distracted_docs))

    if len(distracted_docs) == 0:
        # Fall back to keyword-based filtering
        filtered_docs = list(filtered_docs)
        has_attack_keywords = []
        for doc in filtered_docs:
            check = False
            words = doc2text[doc].split(' ')
            for w in words:
                if language == 'en': attack_keywords = ATTACK_EN_KEYWORDS
                if language == 'es': attack_keywords = ATTACK_ES_KEYWORDS
                for keyword in attack_keywords:
                    if keyword.lower() in w.lower():
                        check = True
                        break
            has_attack_keywords.append(int(check))
        # Check if majority has attack keywords
        ctx_1 = has_attack_keywords.count(1)
        if ctx_1 >= len(filtered_docs) / 2:
            distracted_docs = set()
            for ix in range(len(has_attack_keywords)):
                if not has_attack_keywords[ix]:
                    distracted_docs.add(filtered_docs[ix])
            filtered_docs = all_doc_ids - distracted_docs

    return filtered_docs, distracted_docs
