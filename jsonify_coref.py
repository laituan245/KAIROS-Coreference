import json
from os import listdir
from os.path import isfile, join

MAIN_FILES = ['entity.cs', 'event.cs', 'relation.cs', 'clusters.txt', 'distractors.txt']

def _jsonify(dir):
    data = {}
    files = [f for f in listdir(dir) if isfile(join(dir, f)) and f in MAIN_FILES]
    for f in files:
        with open(join(dir, f), 'r') as output_f:
            content = output_f.read()
        data[f] = content
    return data

def jsonify_coref(coref_output):
    data = _jsonify(coref_output)
    data['en'] = _jsonify(join(coref_output, 'en'))
    data['es'] = _jsonify(join(coref_output, 'en'))
    return json.dumps(data)
