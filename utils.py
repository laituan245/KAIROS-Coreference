import os
import math
import torch
import json
import pyhocon
import numpy as np
import tempfile

from constants import *
from transformers import *
from nltk import word_tokenize
from data import EventCentricDocument
from data.helpers import divide_event_docs
from models import EventCorefModel, EntityCorefModel
from boltons.iterutils import pairwise, windowed

def read_sent_level_event_extraction_input(fp):
    all_words, all_mentions, words_offset = [], [], 0
    sent_lens = []
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    # Process input data
    with open(fp, 'r') as f:
        for line in f:
            data = json.loads(line)
            _id, _sentence, events = data['id'], data['sentence'], data['events']
            # Extract triggers
            triggers, appeared = [], set()
            for _e in events:
                e = _e['trigger']
                e[0], e[1] = int(e[0]), int(e[1])
                should_skip = False
                for _a in appeared:
                    if get_overlap(_a, (e[0], e[1]-1)) > 0:
                        should_skip = True
                if should_skip: continue
                triggers.append({
                    'start': int(e[0]),
                    'end': int(e[1]),     # The end index is non-inclusive.
                    'type': e[2]
                })
                appeared.add((e[0], e[1]-1))
            triggers.sort(key=lambda x: (x.get('start'), x.get('end')))
            # Sanity checks
            appeared = list(appeared)
            for i in range(len(appeared)):
                for j in range(i+1, len(appeared)):
                    assert(get_overlap(appeared[i], appeared[j]) == 0)
            # Divide _sentence into spans
            if len(triggers) == 0:
                spans = [_sentence]
                span_types = [{'type': NOT_ENTITY}]
            else:
                spans, span_types, cur_idx = [], [], 0
                for trigger in triggers:
                    spans.append(_sentence[cur_idx:trigger['start']])
                    span_types.append({
                        'type': NOT_ENTITY,
                        'start_char': None,
                        'end_char': None
                    })
                    spans.append(_sentence[trigger['start']:trigger['end']])
                    span_types.append({
                        'type': trigger['type'],
                        'start_char': trigger['start'],
                        'end_char': trigger['end'],
                        'sent_id': _id
                    })
                    cur_idx = trigger['end']
                if cur_idx < len(_sentence)-1:
                    spans.append(_sentence[cur_idx:])
                    span_types.append({'type': NOT_ENTITY})
            spans = [word_tokenize(span) for span in spans]
            # Create an EventCentricDocument
            word_index = 0
            mentions, words = [], flatten(spans)
            for span, span_type in zip(spans, span_types):
                if span_type['type'] != NOT_ENTITY:
                    start_index = word_index
                    end_index = word_index + len(span)
                    span_type['start'] = words_offset + start_index
                    span_type['end'] = words_offset + end_index
                    mentions.append(span_type)
                word_index += len(span)
            words_offset += len(words)
            all_mentions.extend(mentions)
            all_words.extend(words)
            sent_lens.append(len(words))

    # Splitting
    splitted_docs = divide_event_docs(all_words, all_mentions, sent_lens)

    return splitted_docs 

def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def read_event_types(fp):
    types = {}
    with open(fp, 'r') as f:
        for line in f:
            es = line.split('\t')
            type_name = es[1] + '.' + es[3] + '.' + es[5]
            template = es[8]
            unfiltered_args = es[9:]
            args, arg_ctx = {}, 1
            for i in range(0, len(unfiltered_args), 3):
                if len(unfiltered_args[i].strip()) == 0: continue
                args[unfiltered_args[i]] = '<arg{}>'.format(arg_ctx)
                arg_ctx += 1
            types[type_name] = {
                'type_name': type_name,
                'template': template,
                'args': args
            }
    return types

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def flatten(l):
    return [item for sublist in l for item in sublist]

def load_configs(config_name):
    configs = pyhocon.ConfigFactory.parse_file(BASIC_CONF_PATH)[config_name]
    return configs

def load_tokenizer_and_model(model_type):
    print('Loading model {}'.format(model_type))
    assert(model_type in MODEL_TYPES)
    # Load configs
    if model_type == EVENT_MODEL:
        configs = load_configs(EVENT_COREF_CONFIG)
        saved_path = PRETRAINED_EVENT_MODEL
    if model_type == EN_ENTITY_MODEL:
        configs = load_configs(EN_ENTITY_COREF_CONFIG)
        saved_path = PRETRAINED_EN_ENTITY_MODEL
    if model_type == ES_ENTITY_MODEL:
        configs = load_configs(ES_ENTITY_COREF_CONFIG)
        saved_path = PRETRAINED_ES_ENTITY_MODEL
    print('Loaded configs')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    print('Loaded tokenizer')

    # Load model
    if model_type in [EN_ENTITY_MODEL, ES_ENTITY_MODEL]: model = EntityCorefModel(configs)
    if model_type in [EVENT_MODEL]: model = EventCorefModel(configs)
    print('Initialized model')
    assert(os.path.exists(saved_path))
    if os.path.exists(saved_path):
        checkpoint = torch.load(saved_path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reloaded model from pretrained ckpt')

    return tokenizer, model

def extract_input_masks_from_mask_windows(mask_windows):
    input_masks = []
    for mask_window in mask_windows:
        subtoken_count = listRightIndex(mask_window, -3) + 1
        input_masks.append([1] * subtoken_count + [0] * (len(mask_window) - subtoken_count))
    input_masks = np.array(input_masks)
    return input_masks


def get_predicted_antecedents(antecedents, antecedent_scores, return_scores=False):
    predicted_antecedents, predicted_scores = [], []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
        if index < 0:
            predicted_antecedents.append(-1)
            predicted_scores.append(0)
        else:
            predicted_antecedents.append(antecedents[i, index])
            predicted_scores.append(antecedent_scores[i, index+1])
    if return_scores:
        return predicted_antecedents, predicted_scores
    return predicted_antecedents

def convert_to_sliding_window(expanded_tokens, sliding_window_size, tokenizer):
    """
    construct sliding windows, allocate tokens and masks into each window
    :param expanded_tokens:
    :param sliding_window_size:
    :return:
    """
    CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])
    SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])
    PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])
    expanded_masks = [1] * len(expanded_tokens)
    sliding_windows = construct_sliding_windows(len(expanded_tokens), sliding_window_size - 2)
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window
    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = expanded_tokens[window_start: window_end]
        original_masks = expanded_masks[window_start: window_end]
        window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = CLS + original_tokens + SEP + PAD * (sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
        assert len(one_window_token) == sliding_window_size
        assert len(one_window_mask) == sliding_window_size
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)
    return token_windows, mask_windows

def construct_sliding_windows(sequence_length: int, sliding_window_size: int):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows

def print_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
