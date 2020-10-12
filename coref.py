import os
import time
import torch
import random

from constants import *
from os.path import join
from argparse import ArgumentParser
from entity_coref import entity_coref
from event_coref import event_coref
from utils import create_dir_if_not_exist
from scripts import align_relation, align_event, string_repr

ONEIE = 'oneie'

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--oneie_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/en/oneie/m1')
    parser.add_argument('--linking_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/en/linking/en.linking.wikidata.cs')
    parser.add_argument('--coreference_output', default='/shared/nas/data/m1/tuanml2/tmpfile/docker-compose/output/en/coref/')
    parser.add_argument('--language', default='en')
    args = parser.parse_args()
    assert(args.language in ['en', 'es'])

    # Run entity coref
    entity_cs = join(args.oneie_output, 'cs/entity.cs')
    json_dir = join(args.oneie_output, 'json')
    output_entity =  join(args.coreference_output, 'entity.cs')
    entity_coref(entity_cs, json_dir, args.linking_output, output_entity, args.language)

    # Run event coref
    event_cs = join(args.oneie_output, 'cs/event.cs')
    json_dir = join(args.oneie_output, 'json')
    output_event = join(args.coreference_output, 'event.cs')
    event_coref(event_cs, json_dir, output_event, args.language)

    # Run aligning relation
    input_relation = join(args.oneie_output, 'cs/relation.cs')
    output_relation = join(args.coreference_output, 'relation.cs')
    align_relation(entity_cs, output_entity, input_relation, output_relation)

    # Run aligning event
    align_event(output_entity, output_event)

    # Run string_repr
    string_repr(output_entity, output_event)
