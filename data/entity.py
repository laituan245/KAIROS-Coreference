import os

from utils import *
from os import listdir
from os.path import isfile, join

class EntityCentricDocument:
    def __init__(self, doc_id, words, entity_mentions, tokenizer):
        self.doc_id = doc_id
        self.words = words
        self.entity_mentions = entity_mentions

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
        assert(len(self.word_starts_indexes) == len(self.words))
        assert(len(self.word_ends_indexes) == len(self.words))

        # Build token_windows, mask_windows, and input_masks
        doc_token_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        self.token_windows, self.mask_windows = \
            convert_to_sliding_window(doc_token_ids, 512, tokenizer)
        self.input_masks = extract_input_masks_from_mask_windows(self.mask_windows)

        # Compute gold_starts, gold_ends
        gold_starts, gold_ends = [], []
        for e in entity_mentions:
            e['start'] = self.word_starts_indexes[e['start_token']]
            e['end'] = self.word_ends_indexes[e['end_token']]
            gold_starts.append(e['start'])
            gold_ends.append(e['end'])
        self.gold_starts = np.array(gold_starts)
        self.gold_ends = np.array(gold_ends)

class EntityCentricDataset:
    def __init__(self, data):
        self.data = data
        self.examples, self.tensorized_examples = {}, {}
        self.examples[TEST], self.tensorized_examples[TEST] = [], []

        for inst in data:
            self.examples[TEST].append({
                'doc_key': inst.doc_id,
            })
            self.tensorized_examples[TEST].append(
                (np.array(inst.token_windows), np.array(inst.input_masks), False,
                inst.gold_starts, inst.gold_ends, np.array([]), np.array(inst.mask_windows))
            )
