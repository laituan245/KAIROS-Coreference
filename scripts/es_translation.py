import os
import time
import torch
import random
import shutil

from os.path import dirname, join
from utils import create_dir_if_not_exist
from transformers import MarianMTModel, MarianTokenizer

# Translate from spanish to english
def translate(model, tokenizer, text):
    with torch.no_grad():
        src_text = [
            text
        ]
        translated = model.generate(**tokenizer.prepare_translation_batch(src_text))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return tgt_text[0]


def es_translation(es_linking_path):
    es_linking_cs = join(es_linking_path, 'es.linking.wikidata.cs')

    model_name = 'Helsinki-NLP/opus-mt-es-en'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    lines, en2es = [], {}
    with open(es_linking_cs, 'r') as f:
        for line in f:
            es = line.split('\t')
            if len(es) <= 3:
                lines.append(line.strip())
                continue
            else:
                text = es[2][1:-1]
                if not text in en2es:
                    en2es[text] = translate(model, tokenizer, text)
                text = en2es[text]
                text = es[2][0] + text + es[2][-1]
                es[2] = text
                line = '\t'.join(es)
                lines.append(line)

    with open(es_linking_cs, 'w+') as f:
        for line in lines:
            f.write('{}\n'.format(line))
