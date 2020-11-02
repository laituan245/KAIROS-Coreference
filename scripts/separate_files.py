import os
import time
import torch
import random
import shutil

from os.path import dirname, join
from utils import create_dir_if_not_exist

def read_cs(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        # Filter out modality lines
        filtered_lines = []
        for l in lines:
            es = l.strip().split('\t')
            if es[1] == 'modality': continue
            filtered_lines.append(l)
        lines = filtered_lines
        return lines

def separate_files(entity_output, event_output, relation_output, english_docs, spanish_docs):
    entities_lines = read_cs(entity_output)
    events_lines = read_cs(event_output)
    relations_lines = read_cs(relation_output)

    output_paths = [dirname(entity_output), dirname(event_output), dirname(relation_output)]
    lines_set = [entities_lines, events_lines, relations_lines]
    names = ['entity', 'event', 'relation']

    for name, lines, output_path in zip(names, lines_set, output_paths):
        for language in ['en', 'es']:
            doc_set = english_docs if language == 'en' else spanish_docs
            output_lines = []
            for line in lines:
                es = line.strip().split('\t')
                if len(es) <= 4:
                    output_lines.append(line)
                else:
                    doc_id = es[-2].split(':')[0]
                    if doc_id in doc_set:
                        output_lines.append(line)
            # Refine output_lines
            filtered_lines, should_insert_next_line = [], False
            for i in range(len(output_lines)-1):
                if should_insert_next_line:
                    filtered_lines.append(output_lines[i])
                    continue
                line = output_lines[i]
                next_line = output_lines[i+1]
                es = line.strip().split('\t')
                next_es = next_line.strip().split('\t')
                should_insert_next_line = False
                if len(es) > 3:
                    filtered_lines.append(line)
                    if i == len(output_lines) - 2: should_insert_next_line = True
                    continue
                if name == 'entity':
                    if len(next_es) <= 3 and es[1] == 'type' and next_es[1] == 'link':
                        should_insert_next_line = False
                        continue
                    if es[1] == 'link':
                        prev_es = output_lines[i-1].strip().split('\t')
                        if len(prev_es) <= 3 and prev_es[1] == 'type':
                            continue
                    filtered_lines.append(line)
                    if i == len(output_lines) - 2: should_insert_next_line = True
                if name == 'event':
                    if len(next_es) <= 3:
                        should_insert_next_line = False
                        continue
                    filtered_lines.append(line)
                if i == len(output_lines) - 2: should_insert_next_line = True
            if should_insert_next_line: filtered_lines.append(output_lines[-1])
            # Output
            base_output_path = join(output_path, language)
            create_dir_if_not_exist(base_output_path)
            with open(join(base_output_path, '{}.cs'.format(name)), 'w+') as f:
                for line in filtered_lines:
                    f.write('{}\n'.format(line.strip()))
