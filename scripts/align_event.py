import os
import time
import torch
import random

from argparse import ArgumentParser

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--input_entity', default='/shared/nas/data/m1/tuanml2/kairos/output/corrected_format/entity.cs')
    parser.add_argument('--input_event', default='/shared/nas/data/m1/tuanml2/kairos/output/corrected_format/event.cs')
    parser.add_argument('--output_path', default='/shared/nas/data/m1/tuanml2/kairos/output/corrected_format/event_2.cs')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # DEBUG Mode
    if args.debug:
        args.input_entity = 'resources/samples/outputs/entity.cs'
        args.input_event = 'resources/samples/outputs/event.cs'
        args.output_path = 'resources/samples/outputs/new_event.cs'

    # Read output entity file
    mid2eid = {}
    with open(args.input_entity, 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            es = line.split('\t')
            if es[1].endswith('mention'):
                mid2eid[es[-2]] = es[0]

    # Output file
    with open(args.output_path, 'w+', encoding='utf-8') as output_f:
        with open(args.input_event, 'r', encoding='utf-8') as event_f:
            for line in event_f:
                line = line.strip()
                if len(line) == 0: continue
                es = line.split('\t')
                if es[1].startswith('type') or es[1].startswith('mention.actual') or es[1].startswith('canonical_mention.actual'):
                    output_f.write('{}\n'.format(line))
                else:
                    mid = es[-2]
                    es[-3] = mid2eid[mid]
                    output_f.write('{}\n'.format('\t'.join(es)))

    # Delete old file
    os.remove(args.input_event)
    os.rename(args.output_path, args.input_event)
