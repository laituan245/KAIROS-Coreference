import os
import time
import torch
import random

from argparse import ArgumentParser

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--original_input_entity', default='/shared/nas/data/m1/yinglin8/kairos/result/all_oct_4/m1_m2/cs/entity.cs')
    parser.add_argument('--new_input_entity', default='/shared/nas/data/m1/tuanml2/kairos/output/corrected_format/entity.cs')
    parser.add_argument('--input_relation', default='/shared/nas/data/m1/yinglin8/kairos/result/all_oct_4/m1_m2/cs/relation.cs')
    parser.add_argument('--output_relation', default='/shared/nas/data/m1/tuanml2/kairos/output/corrected_format/relation.cs')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # DEBUG Mode
    if args.debug:
        args.original_input_entity = 'resources/samples/entity.cs'
        args.new_input_entity = 'resources/samples/outputs/entity.cs'
        args.input_relation = 'resources/samples/outputs/relation.cs'
        args.output_relation = 'resources/samples/outputs/new_relation.cs'

    # Read (old) entity file
    olde2mid = {}
    with open(args.original_input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            es = line.split('\t')
            if es[1].startswith('mention') or es[1].startswith('canonical_mention'):
                olde2mid[es[0]] = es[-2]

    # Read (new) entity file
    mid2eid = {}
    with open(args.new_input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            es = line.split('\t')
            if es[1].endswith('mention'):
                mid2eid[es[-2]] = es[0]

    # Output file
    with open(args.output_relation, 'w+', encoding='utf-8') as output_f:
        with open(args.input_relation, 'r', encoding='utf-8') as event_f:
            for line in event_f:
                line = line.strip()
                if len(line) == 0: continue
                es = line.split('\t')
                assert(len(es) == 5)
                if not es[-2].startswith('wiki_'): continue
                es[0] = mid2eid[olde2mid[es[0]]]
                es[2] = mid2eid[olde2mid[es[2]]]
                output_f.write('{}\n'.format('\t'.join(es)))
