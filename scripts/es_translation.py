import os
import time
import torch
import random
import shutil
from nltk.tag import pos_tag

from deep_translator import GoogleTranslator
from os.path import dirname, join
from utils import create_dir_if_not_exist

# Translate from spanish to english
def translate(text):
    if text == '%': return '%'
    translated = GoogleTranslator(source='es', target='en').translate(text)
    return translated

def read_en_linking_cs(en_linking_cs):
    english_mentions = set()
    with open(en_linking_cs, 'r') as f:
        for line in f:
            es = line.split('\t')
            if len(es) <= 3:
                continue
            english_mentions.add(es[2][1:-1].lower())
    return english_mentions

def es_translation(es_linking_path, en_linking_path):
    en_linking_cs = join(en_linking_path, 'en.linking.wikidata.cs')
    es_linking_cs = join(es_linking_path, 'es.linking.wikidata.cs')

    # Read en linking cs
    english_mentions = read_en_linking_cs(en_linking_cs)

    lines, en2es = [], {}
    with open(es_linking_cs, 'r') as f:
        for line in f:
            es = line.split('\t')
            if len(es) <= 3:
                lines.append(line)
                continue
            else:
                text = es[2][1:-1]
                if text.lower() in english_mentions:
                    en2es[text] = text
                if not text in en2es:
                    translation = translate(text)
                    if translation.lower().strip() == text.lower().strip():
                        translation = text
                    en2es[text] = translation
                text = en2es[text]
                text = es[2][0] + text + es[2][-1]
                es[2] = text
                line = '\t'.join(es)
                lines.append(line)

    with open(es_linking_cs, 'w+') as f:
        for line in lines:
            f.write('{}'.format(line))

    return en2es
