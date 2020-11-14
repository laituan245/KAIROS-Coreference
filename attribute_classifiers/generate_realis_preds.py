import os
import torch
import pyhocon
import json

from data import *
from utils import *
from os.path import join
from transformers import *
from modality_classifier_models import BaseAttributesClassifier

# Constants
CONFIG_NAME = 'kbp_realis_classifier'
TRAINED_MODEL = 'realis_classifier_model.pt'
OUTPUT_FILE = 'attrs_preds.json'
REALIS_TYPES = ['actual', 'generic', 'other']

def generate_realis_preds(cs_path, json_dir, output_path):
    create_dir_if_not_exist(dirname(output_path))
    output_file_path = join(output_path, OUTPUT_FILE)

    # Read configs andtoad tokenizer and model
    configs = pyhocon.ConfigFactory.parse_file('configs/basic.conf')[CONFIG_NAME]
    device = torch.device('cuda' if torch.cuda.is_available() and not configs['no_cuda'] else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    model = BaseAttributesClassifier(configs, device, REALIS_TYPES)
    assert(os.path.exists(TRAINED_MODEL))
    if os.path.exists(TRAINED_MODEL):
        checkpoint = torch.load(TRAINED_MODEL, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Reloaded the model')

    docs = load_event_centric_dataset(tokenizer, cs_path, json_dir)

    # Apply model for prediction
    loc2preds = {}
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as json_file:
            loc2preds = json.load(json_file)
    print('Number of instances: {}'.format(len(docs)))

    with torch.no_grad():
        for inst in docs:
            event_mentions = inst.event_mentions
            if len(event_mentions) == 0: continue
            preds, _ = model(inst, is_training=False)
            for ix, e in enumerate(event_mentions):
                e_id = e['mention_id']
                if not e_id in loc2preds: loc2preds[e_id] = {}
                loc2preds[e_id]['event_realis'] = preds[ix]

    with open(output_file_path, 'w+') as outfile:
        json.dump(loc2preds, outfile)

    model.to(torch.device('cpu'))
