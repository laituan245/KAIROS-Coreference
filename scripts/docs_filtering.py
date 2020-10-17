import os
import time
import torch
import random

from os.path import dirname
from transformers import *
from data import load_event_centric_dataset

EN_KEYWORDS = ['bomb', 'bombs', 'explosive', 'explosives', 'drone', 'drones',
               'strike', 'strikes', 'attack', 'attacks']
ES_KEYWORDS = []

def docs_filtering(event_cs_path, json_dir, language):
    if language == 'es':
        assert(len(ES_KEYWORDS) > 0)
    tokenizer = AutoTokenizer.from_pretrained('SpanBERT/spanbert-large-cased', do_basic_tokenize=False) # Not really matter
    docs = load_event_centric_dataset(tokenizer, event_cs_path, json_dir)

    all_docs, filtered_docs = set(), set()
    for doc in docs:
        should_keep = True

        # Heuristics rule
        words = doc.words
        has_keywords = False
        for w in words:
            w = w.lower()
            if language == 'en': KEYWORDS = EN_KEYWORDS
            if language == 'es': KEYWORDS = ES_KEYWORDS
            for keyword in KEYWORDS:
                if keyword in w:
                    has_keywords = True
        if not has_keywords: should_keep = False

        # Update filtered_docs
        doc_id = doc.doc_id
        doc_id = doc_id[:doc_id.find('_part')]
        if should_keep:
            filtered_docs.add(doc_id)
        all_docs.add(doc_id)
    distracted_docs = all_docs - filtered_docs

    print('[AFTER FILTERING] Remaining doc ids: {}'.format(filtered_docs))

    return filtered_docs, distracted_docs
