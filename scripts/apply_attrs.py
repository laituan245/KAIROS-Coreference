import nltk
import json

from data import *
from utils import *
from os.path import join
from transformers import *

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum[0]

def apply_attrs(output_path):
    # Load predicted attributes
    loc2preds = {}
    attrs_preds_file = join(output_path, 'attrs_preds.json')
    with open(attrs_preds_file, 'r') as json_file:
        loc2preds = json.load(json_file)

    # Read output event.cs file
    all_lines = []
    event2hedge, event2polarity, event2realis = {}, {}, {}
    output_event_cs = join(output_path, 'event.cs')
    with open(output_event_cs, 'r') as f:
        for line in f:
            all_lines.append(line)
            es = line.strip().split('\t')
            if 'mention' in es[1]:
                event_id = es[0]
                mention_id = es[3]
                if mention_id in loc2preds:
                    if not event_id in event2hedge: event2hedge[event_id] = []
                    if not event_id in event2polarity: event2polarity[event_id] = []
                    if not event_id in event2realis: event2realis[event_id] = []
                    preds = loc2preds[mention_id]
                    event2hedge[event_id].append(preds['event_hedge'])
                    event2polarity[event_id].append(preds['event_polarity'])
                    event2realis[event_id].append(preds['event_realis'])

    # Find majority
    for e in event2hedge:
        hedge_values = event2hedge[e]
        if hedge_values.count('C') / len(hedge_values) > 0.5:
            event2hedge[e] = 'C'
        else:
            event2hedge[e] = find_majority(event2hedge[e])

        event2polarity[e] = find_majority(event2polarity[e])

        realis_values = event2realis[e]
        if realis_values.count('actual') / len(realis_values) > 0.5:
            event2realis[e] = 'actual'
        else:
            event2realis[e] = find_majority(event2realis[e])

    # Output
    with open(output_event_cs, 'w+') as f:
        for line in all_lines:
            f.write('{}\n'.format(line.strip()))
            es = line.strip().split('\t')
            if es[1] == 'type':
                # Write attribute line
                attr_es = [es[0], 'modality', '']
                if es[0] in event2hedge:
                    values = []
                    if event2realis[es[0]] == 'generic': values.append('generic')
                    if event2hedge[es[0]] == 'U' and not event2realis[es[0]] == 'actual': values.append('hedged')
                    if event2realis[es[0]] == 'other': values.append('irrealis')
                    if event2polarity[es[0]] == 'Negative': values.append('negated')
                    if len(values) == 0: values.append('actual')
                    attr_es[-1] = ','.join(values)
                else:
                    attr_es[-1] = 'actual'

                modality_line = '\t'.join(attr_es)
                f.write('{}\n'.format(modality_line.strip()))
