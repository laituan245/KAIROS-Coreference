import os
import time
import json
import torch
import random
import shutil
from nltk.tag import pos_tag

from os.path import dirname, join
from utils import create_dir_if_not_exist
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import MarianMTModel, MarianTokenizer
DetectorFactory.seed = 0

def is_english(text):
    try:
        if detect(text) != "en":
            return False
    except LangDetectException:
        return False
    return True

# Translate from spanish to english
def translate(model, tokenizer, text):
    with torch.no_grad():
        src_text = [
            text
        ]
        translated = model.generate(**tokenizer.prepare_translation_batch(src_text))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return tgt_text[0]

def translate_extensions(extension_fp, output_fp=None):
    if output_fp is None:
        output_fp = extension_fp

    # Prepare translation model
    model_name = 'Helsinki-NLP/opus-mt-es-en'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Read original extension file
    with open(extension_fp, 'r') as f:
        data = json.loads(f.read())

    # Translating ...
    new_data = {}
    for k in data:
        [original_mention, extend_mention, extend_offset] = data[k]
        if is_english(original_mention):
            en_mention = original_mention
        else:
            en_mention = translate(model, tokenizer, original_mention)
        if is_english(extend_mention):
            en_extend_mention = extend_mention
        else:
            en_extend_mention = translate(model, tokenizer, extend_mention)
        new_data[k] = [en_mention, en_extend_mention, extend_offset]

    # Write output
    with open(output_fp, 'w+') as f:
        f.write(json.dumps(new_data))
