import os
import copy

from utils import *
from os import listdir
from os.path import isfile, join

class EntityCentricDocument:
    def __init__(self, doc_id, words, entity_mentions, tokenizer):
        self.doc_id = doc_id
        self.words = words
        self.entity_mentions = entity_mentions
        self.num_words = len(self.words)

        # Build doc_tokens, self.word_starts_indexes, self.word_ends_indexes
        doc_tokens, word_starts_indexes, word_ends_indexes, start_index = [], [], [], 0
        for w in self.words:
            word_tokens = tokenizer.tokenize(w)
            doc_tokens += word_tokens
            word_starts_indexes.append(start_index)
            word_ends_indexes.append(start_index + len(word_tokens)-1)
            start_index += len(word_tokens)
        self.word_starts_indexes = word_starts_indexes
        self.word_ends_indexes = word_ends_indexes
        self.doc_tokens = doc_tokens
        assert(len(self.word_starts_indexes) == len(self.words))
        assert(len(self.word_ends_indexes) == len(self.words))

        # Update start and end fields of entity mentions
        for e in entity_mentions:
            e['start'] = self.word_starts_indexes[e['start_token']]
            e['end'] = self.word_ends_indexes[e['end_token']]

class EntityCentricDocumentPair:
    def __init__(self, doc1, doc2, tokenizer):
        self.doc1 = doc1
        self.doc2 = doc2

        # All words
        words = self.words = doc1.words + doc2.words
        self.num_words = len(self.words)

        # All entity mentions
        entity_mentions_1 = copy.deepcopy(doc1.entity_mentions)
        entity_mentions_2 = copy.deepcopy(doc2.entity_mentions)
        for m in entity_mentions_2:
            m['start'] += len(doc1.doc_tokens)
            m['end'] += len(doc1.doc_tokens)
        self.entity_mentions = entity_mentions_1 + entity_mentions_2

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

        # Build gold_starts, gold_ends
        gold_starts, gold_ends = [], []
        for e in self.entity_mentions:
            gold_starts.append(e['start'])
            gold_ends.append(e['end'])
        self.gold_starts = np.array(gold_starts)
        self.gold_ends = np.array(gold_ends)

        # tensorized_example
        self.tensorized_example = (
            (np.array(self.token_windows), np.array(self.input_masks), False,
            self.gold_starts, self.gold_ends, np.array([]), np.array(self.mask_windows))
        )

        # Sanity test
        # for ix, e in enumerate(self.entity_mentions):
        #     first_token, last_token = doc_tokens[self.gold_starts[ix]], doc_tokens[self.gold_ends[ix]]
        #     first_token = first_token.replace('##', '')
        #     last_token = last_token.replace('##', '')
        #     assert(e['canonical_mention'].startswith(first_token))
        #     assert(e['canonical_mention'].endswith(last_token))
