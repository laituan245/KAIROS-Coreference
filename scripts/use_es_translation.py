import nltk
import json

from data import *
from utils import *
from os.path import join
from transformers import *

def use_es_translation(linking_path):
    # Translation file
    mid2translation = {}
    translation_file = join(linking_path, 'es.linking.wikidata.tab')
    with open(translation_file, 'r') as f:
        for line in f:
            es = line.strip().split('\t')
            assert(len(es) == 8 or len(es) == 9)
            if len(es) == 8: continue
            mention_id = es[3]
            translation = es[-1].split('|')[0]
            mid2translation[mention_id] = translation

    # Update es.linking.wikidata.cs
    all_lines = []
    main_file = join(linking_path, 'es.linking.wikidata.cs')
    with open(main_file, 'r') as f:
        for line in f:
            all_lines.append(line)

    with open(main_file, 'w+') as f:
        for line in all_lines:
            es = line.strip().split('\t')
            if len(es) <= 3:
                f.write(line)
                continue
            if 'mention' in es[1]:
                mention_id = es[3]
                if not mention_id in mid2translation:
                    f.write(line)
                else:
                    translation = mid2translation[mention_id]
                    es[2] = '"{}"'.format(translation)
                    line = '\t'.join(es)
                    f.write('{}\n'.format(line))
                continue
            f.write(line)
