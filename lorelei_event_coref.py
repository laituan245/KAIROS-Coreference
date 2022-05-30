import argparse

from utils import *
from data import EventCentricDocument
from data.helpers import divide_event_docs
from constants import EVENT_MODEL
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fp', default='resources/30may/0206_args.json')
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(EVENT_MODEL)
    print('Loaded tokenizer and model')

    # Data preparation
    all_words, all_mentions, sent_lens = read_sent_level_event_extraction_input(args.input_fp)
    raw_docs = divide_event_docs(all_words, all_mentions, sent_lens)
    # Update test_docs
    test_docs = []
    for ix, (cur_words, cur_mentions) in enumerate(raw_docs):
        test_doc = EventCentricDocument('part_{}'.format(ix), cur_words, cur_mentions)
        test_doc.tokenize(tokenizer)
        test_docs.append(test_doc)
    test_docs.sort(key=lambda x: x.doc_id)

    # Info
    doc_ids = [x.doc_id for x in test_docs]
    print('Number of splitted docs: {}'.format(len(doc_ids)))


