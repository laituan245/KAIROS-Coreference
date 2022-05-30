import nltk
import copy

from utils import *

class EventCentricDocument:
    def __init__(self, doc_id, words, event_mentions):
        self.doc_id = doc_id
        self.words = words
        self.event_mentions = event_mentions
        self.num_words = len(self.words)

        # Sort by trigger start
        self.event_mentions.sort(key=lambda x: x['start'])

        # Default values
        self.word_starts_indexes, self.doc_tokens = [], []

    def tokenize(self, tokenizer):
        # Tokenization
        doc_tokens, word_starts_indexes, start_index = [], [], 0
        for w in self.words:
            word_tokens = tokenizer.tokenize(w)
            doc_tokens += word_tokens
            word_starts_indexes.append(start_index)
            start_index += len(word_tokens)
        self.word_starts_indexes = word_starts_indexes
        self.doc_tokens = doc_tokens
        assert(len(self.word_starts_indexes) == len(self.words))

        # Build token_windows, mask_windows, and input_masks
        doc_token_ids = tokenizer.convert_tokens_to_ids(self.doc_tokens)
        self.token_windows, self.mask_windows = \
            convert_to_sliding_window(doc_token_ids, 512, tokenizer)
        self.input_masks = extract_input_masks_from_mask_windows(self.mask_windows)

class EventCentricDocumentPair:
    def __init__(self, doc1, doc2, tokenizer):
        self.doc1 = doc1
        self.doc2 = doc2

        # All words
        words = self.words = doc1.words + doc2.words
        self.num_words = len(self.words)

        # All event mentions
        event_mentions_1 = copy.deepcopy(doc1.event_mentions)
        event_mentions_2 = copy.deepcopy(doc2.event_mentions)
        for m in event_mentions_2:
            m['start'] += len(doc1.words)
            m['end'] += len(doc1.words)
        self.event_mentions = event_mentions_1 + event_mentions_2

        # Sanity test
        # for e in self.event_mentions:
        #     first_word, last_word = words[e['start']], words[e['end']-1]
        #     assert(e['text'].startswith(first_word))
        #     assert(e['text'].endswith(last_word))

        # Build doc_tokens
        self.doc_tokens = doc_tokens = doc1.doc_tokens + doc2.doc_tokens
        word_starts_indexes_1 = copy.deepcopy(doc1.word_starts_indexes)
        word_starts_indexes_2 = copy.deepcopy(doc2.word_starts_indexes)
        for i in range(len(word_starts_indexes_2)):
            word_starts_indexes_2[i] += len(doc1.doc_tokens)
        self.word_starts_indexes = word_starts_indexes_1 + word_starts_indexes_2
        assert(len(self.word_starts_indexes) == self.num_words)

        # Build token_windows, mask_windows, and input_masks
        doc_token_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        self.token_windows, self.mask_windows = \
            convert_to_sliding_window(doc_token_ids, 512, tokenizer)
        self.input_masks = extract_input_masks_from_mask_windows(self.mask_windows)
