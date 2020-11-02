import nltk
import json
nltk.download('treebank')
nltk.download('conll2000')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from data import *
from utils import *
from transformers import *
from uncertainty.classifier import Classifier

OUTPUT_FILE = 'attrs_preds.json'

def generate_hedge_preds(cs_path, json_dir, output_path):
    output_file_path = join(output_path, OUTPUT_FILE)
    create_dir_if_not_exist(dirname(output_path))
    classifier = Classifier(granularity='sentence', binary=True)
    tokenizer = BertTokenizer.from_pretrained('SpanBERT/spanbert-large-cased', do_basic_tokenize=False)
    docs = load_event_centric_dataset(tokenizer, cs_path, json_dir)

    # Apply model for prediction
    loc2preds = {}
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as json_file:
            loc2preds = json.load(json_file)

    for doc in docs:
        words, event_mentions = doc.words, doc.event_mentions
        for ev in event_mentions:
            mention_id = ev['mention_id']
            ev_content = doc.words[ev['start']:ev['end']]
            context_left = doc.words[ev['start']-8:ev['start']]
            context_right = doc.words[ev['end']:ev['end']+8]
            sentence = ' '.join(context_left + ev_content + context_right)
            prediction = classifier.predict(sentence)
            if not mention_id in loc2preds: loc2preds[mention_id] = {}
            loc2preds[mention_id]['event_hedge'] = prediction

    with open(output_file_path, 'w+') as outfile:
        json.dump(loc2preds, outfile)
