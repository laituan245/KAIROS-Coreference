import os
import time
import torch
import random
import shutil

from os.path import dirname, join
from utils import create_dir_if_not_exist
from transformers import MarianMTModel, MarianTokenizer

# Translate from english to spanish
def translate(model, tokenizer, text):
    src_text = [
        '>>es<< ' + text
    ]
    translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text[0]


def es_translation(es_linking_path):
    es_linking_cs = join(es_linking_path, 'es.linking.wikidata.cs')

    model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
