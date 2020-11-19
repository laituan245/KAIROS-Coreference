import json
from os.path import join
from data.helpers import read_json_docs
from utils import read_cluster_info

oneie_output =  '/shared/nas/data/m1/tuanml2/oneie_with_relation'
entity_cs = join(oneie_output, 'cs/entity.cs')
json_dir = join(oneie_output, 'json')
event_cs = join(oneie_output, 'cs/event.cs')
json_docs = read_json_docs(json_dir)

clusters = read_cluster_info('resources/original/clusters.txt')

new_docs = set()
doc2cluster = {}
for ix, c in enumerate(clusters):
    for doc_id in c:
        doc2cluster[doc_id] = ix

for doc in json_docs:
    if not doc in doc2cluster:
        new_docs.add(doc)

print('nb of new docs is {}'.format(len(new_docs)))
for doc in new_docs:
    clusters.append([doc])

f = open('resources/original/clusters.txt', 'w+')
for cluster in clusters: f.write('{}\n'.format(json.dumps(cluster)))
f.close()

