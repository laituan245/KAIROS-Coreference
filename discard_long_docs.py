import json

from os.path import join
from utils import read_cluster_info

base_path = 'resources/original'

doc_lengths = json.loads(open(join(base_path, 'doc_lengths.txt')).read())
clusters = read_cluster_info(join(base_path, 'clusters.txt'))

# Decide discard docs
discarded_docs = []
for doc_id, length in doc_lengths:
    if length >= 2000:
        discarded_docs.append(doc_id)
with open('resources/processed/discarded_docs.txt', 'w+') as f:
    for doc_id in discarded_docs:
        f.write('{}\n'.format(doc_id.strip()))
discarded_docs = set(discarded_docs)

# Filtered clusters
filtered_clusters = []
for cluster in clusters:
    _cluster = [c for c in cluster if not c in discarded_docs]
    if len(_cluster) > 0:
        filtered_clusters.append(_cluster)

with open('resources/processed/clusters.txt', 'w+') as f:
    for cluster in filtered_clusters:
        f.write('{}\n'.format(json.dumps(cluster)))
