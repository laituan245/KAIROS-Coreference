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

    # doc_ids, texts, embeddings
    doc_ids, texts, embeddings = [], [], []
    for doc in doc2text:
        doc_ids.append(doc)
        texts.append(doc2text[doc])
    all_doc_ids = set(doc_ids)

    return all_doc_ids, set()  # No filtering
