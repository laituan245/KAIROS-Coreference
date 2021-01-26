import json
import copy

from dateutil.parser import parse

from os import listdir
from os.path import isfile, join
from utils import flatten
from data.event import EventCentricDocument
from data.entity import EntityCentricDocument

def locstr_to_loc(loc_str):
    doc_id, offset_info = loc_str.split(':')
    start, end = offset_info.split('-')
    start, end = int(start), int(end)
    return (doc_id, start, end)

def read_cs(path, run_sanity_checks=True, skip_firstline=False):
    print('Reading cs file {}'.format(path))
    e2info = {}
    with open(path, 'r', encoding='utf-8') as f:
        if skip_firstline: f.readline() # Skip the first line
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            es = line.split('\t')
            e_id = es[0].strip()
            if not e_id in e2info:
                e2info[e_id] = {
                    'mentions': {}
                }
            e = e2info[e_id]
            if es[1] == 'type':
                e['type'] = es[2]
            if es[1].startswith('mention') or es[1].startswith('canonical_mention'):
                loc_str = es[-2]
                if not loc_str in e['mentions']:
                    loc = locstr_to_loc(loc_str)
                    e['mentions'][loc_str] = {
                        'doc_id': loc[0], 'start': loc[1], 'end': loc[2],
                        'mention_id': loc_str
                    }
                if es[1].startswith('mention'): e['mentions'][loc_str]['text'] = es[-3][1:-1]
                if es[1].startswith('canonical_mention'): e['mentions'][loc_str]['canonical_mention'] = es[-3][1:-1]
            if es[1] == 'link':
                e['link'] = es[2]

    if run_sanity_checks:
        ctx, no_text_ctx = 0, 0
        for e in e2info.values():
            for m in e['mentions'].values():
                ctx += 1
                if not 'text' in m:
                    no_text_ctx += 1
                    continue
                #assert(len(m['text']) == m['end'] - m['start'] + 1) # Inclusive Endpoints
        print('Total number of mentions: {}'.format(ctx))
        print('{} mentions do not have text field'.format(no_text_ctx))

    return e2info

def read_json_docs(base_path, filtered_docs = None):
    doc2sents = {}
    for f in listdir(base_path):
        if isfile(join(base_path, f)) and f.endswith('json'):
            file_path = join(base_path, f)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    doc_id = data['doc_id']
                    if not (filtered_docs is None):
                        if not (doc_id in filtered_docs): continue
                    if not doc_id in doc2sents: doc2sents[doc_id] = []
                    sents, tokens, token_ids = [], data['tokens'], data['token_ids']
                    for token, token_id in zip(tokens, token_ids):
                        _doc_id, start, end = locstr_to_loc(token_id)
                        assert(_doc_id == doc_id)
                        sents.append((token, start, end))
                    doc2sents[doc_id].append(sents)
    print('Number of docs from {} is {}'.format(base_path, len(doc2sents)))
    return doc2sents

def divide_event_docs(words, mentions, sent_lens, max_length=1500):
    splitted_docs = []
    start_index, end_index = 0, 0
    sent_lens.append(1000000000000000000) # A hack to include the last splitted doc
    for sent_len in sent_lens:
        cur_length = end_index - start_index
        if cur_length + sent_len <= max_length:
            end_index += sent_len
        else:
            cur_words = copy.deepcopy(words[start_index:end_index])
            cur_mentions = []
            for m in mentions:
                if m['start'] >= start_index and end_index >= m['end']:
                    copied_m = copy.deepcopy(m)
                    copied_m['start'] -= start_index
                    copied_m['end'] -= start_index
                    cur_mentions.append(copied_m)
            start_index = end_index
            splitted_docs.append((cur_words, cur_mentions))
            # Sanity check
            for cur_mention in cur_mentions:
                start_word, end_word = cur_words[cur_mention['start']], cur_words[cur_mention['end']-1]
                assert(cur_mention['text'].startswith(start_word))
                assert(cur_mention['text'].endswith(end_word))

    return splitted_docs

def load_event_centric_dataset(tokenizer, cs_path, json_base_path, filtered_docs = None):
    events = read_cs(cs_path, skip_firstline=False)
    for e in events.values():
        for m in e['mentions'].values():
            m['event_type'] = e['type']
    docs = read_json_docs(json_base_path, filtered_docs)

    # Build doc2mentions
    doc2mentions = {}
    for e in events.values():
        for m in e['mentions'].values():
            doc_id = m['doc_id']
            if not doc_id in docs: continue
            if not doc_id in doc2mentions: doc2mentions[doc_id] = []
            doc2mentions[doc_id].append(m)

    # Build EventCentricDocument
    test_docs = []
    for doc_id in docs:
        sents = docs[doc_id]
        if not doc_id in doc2mentions: continue # No mentions
        tokens = flatten(sents)
        mentions = doc2mentions[doc_id]
        words = [t[0] for t in tokens]
        sent_lens = [len(sent) for sent in sents]

        # Build startchar2token and endchar2token
        startchar2token, endchar2token = {}, {}
        for ix, (_, start, end) in enumerate(tokens):
            startchar2token[start] = ix
            endchar2token[end] = ix

        # Change m['start'] and m['end'] to token indices
        # Append fb_id field
        for m in mentions:
            m['start'] = startchar2token[m['start']]
            m['end'] = endchar2token[m['end']] + 1

        # Divide docs
        splitted_docs = divide_event_docs(words, mentions, sent_lens)

        # Update test_docs
        for ix, (cur_words, cur_mentions) in enumerate(splitted_docs):
            test_doc = EventCentricDocument('{}_part_{}'.format(doc_id, ix), cur_words, cur_mentions)
            test_doc.tokenize(tokenizer)
            test_docs.append(test_doc)

    test_docs.sort(key=lambda x: x.doc_id)

    for i in range(len(test_docs)):
        for j in range(i+1, len(test_docs)):
            di = test_docs[i].doc_id
            dj = test_docs[j].doc_id
            di = parse(di[di.find('__')+2:di.rfind('__')])
            dj = parse(dj[dj.find('__')+2:dj.rfind('__')])
            if di > dj:
                test_docs[i], test_docs[j] = test_docs[j], test_docs[i]

    # Info
    doc_ids = [x.doc_id for x in test_docs]
    print('Number of splitted docs: {}'.format(len(doc_ids)))

    return test_docs

def load_entity_centric_dataset(tokenizer, cs_path, json_base_path, fb_linking_path, filtered_docs = None):
    entities = read_cs(cs_path)
    docs = read_json_docs(json_base_path, filtered_docs)
    linked_entities = read_cs(fb_linking_path, skip_firstline=True)

    # Build mention2type
    mention2fb = {}
    for e in linked_entities.values():
        for mid in e['mentions']:
            mention2fb[mid] = e['link']

    # Build doc2mentions
    doc2mentions = {}
    for e in entities.values():
        for m in e['mentions'].values():
            doc_id = m['doc_id']
            if not doc_id in docs: continue
            if not doc_id in doc2mentions: doc2mentions[doc_id] = []
            doc2mentions[doc_id].append(m)

    # Add field type to each entity
    for e in entities.values():
        for m in e['mentions']:
            e['mentions'][m]['type'] = e['type']

    # Build EntityCentricDocument
    test_docs = []
    for doc_id in docs:
        sents = docs[doc_id]
        if not doc_id in doc2mentions: continue # No mentions
        tokens = flatten(sents)
        mentions = doc2mentions[doc_id]
        words = [t[0] for t in tokens]

        # Build startchar2token and endchar2token
        startchar2token, endchar2token = {}, {}
        for ix, (_, start, end) in enumerate(tokens):
            startchar2token[start] = ix
            endchar2token[end] = ix

        # Change m['start'] and m['end'] to token indices
        # Append fb_id field
        for m in mentions:
            m['start_token'] = startchar2token[m['start']]
            m['end_token'] = endchar2token[m['end']]
            if m['mention_id'] in mention2fb:
                m['fb_id'] = mention2fb[m['mention_id']]
            del m['start']
            del m['end']

        test_docs.append(EntityCentricDocument(doc_id, words, mentions, tokenizer))

    for i in range(len(test_docs)):
        for j in range(i+1, len(test_docs)):
            di = test_docs[i].doc_id
            dj = test_docs[j].doc_id
            di = parse(di[di.find('__')+2:di.rfind('__')])
            dj = parse(dj[dj.find('__')+2:dj.rfind('__')])
            if di > dj:
                test_docs[i], test_docs[j] = test_docs[j], test_docs[i]

    return entities, test_docs
