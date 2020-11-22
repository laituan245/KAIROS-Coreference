import os
import shutil
import argparse
import sys
import itertools
import xml.dom.minidom
import xml.etree.ElementTree as ET
import codecs
import json
import random

from os import listdir
from shutil import rmtree
from os.path import isfile, join
from data import read_json_docs
from scripts import docs_filtering
from utils import create_dir_if_not_exist, flatten

CLUSTERS_FILEPATH = 'resources/filtering_test/clusters.txt'
JSON_DOCS_FILEPATH = '/shared/nas/data/m1/tuanml2/7000_docs/json'
TEST_BASE_PATH = '/shared/nas/data/m1/tuanml2/clustering_tests/'
NB_TESTS = 10
TEST_SIZE = 10
NB_NEGATIVES = 3
NB_POSITIVES = TEST_SIZE - NB_NEGATIVES

random.seed(1995)

# Helper function
def read_cluster_info(cluster_fp):
    clusters = []
    with open(cluster_fp, 'r') as f:
        for line in f:
            clusters.append(json.loads(line))
    return clusters

def get_doc_ids(json_base_path):
    fns = [f for f in listdir(json_base_path) if isfile(join(json_base_path, f))]
    fns = [f for f in fns if f.endswith('.json')]
    doc_ids = [f[:f.find('.json')] for f in fns]
    return doc_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--creating_testcases', action='store_true')
    args = parser.parse_args()

    # Extract clustering information
    doc2cluster = {}
    clusters = read_cluster_info(CLUSTERS_FILEPATH)
    for ix, c in enumerate(clusters):
        for doc_id in c:
            doc2cluster[doc_id] = ix

    # Get doc ids
    doc_ids = get_doc_ids(JSON_DOCS_FILEPATH)

    # Creating testcases
    if not os.path.exists(TEST_BASE_PATH) or args.creating_testcases:
        if os.path.exists(TEST_BASE_PATH): rmtree(TEST_BASE_PATH)
        create_dir_if_not_exist(TEST_BASE_PATH)
        for test_nb in range(NB_TESTS):
            test_path = join(TEST_BASE_PATH, 'test_{}'.format(test_nb+1))
            # Sample positive docs
            while True:
                selected_c = random.choice(clusters)
                if len(selected_c) > NB_POSITIVES:
                    random.shuffle(selected_c)
                    positive_docs = selected_c[:NB_POSITIVES]
                    break
            c_label = doc2cluster[positive_docs[0]]
            # Sample negative docs
            negative_docs = []
            while len(negative_docs) < NB_NEGATIVES:
                random_doc = random.choice(doc_ids)
                if random_doc in doc2cluster and doc2cluster[random_doc] != c_label:
                    negative_docs.append(random_doc)
            all_docs = positive_docs + negative_docs
            # Copy files to directory
            create_dir_if_not_exist(test_path)
            for doc in all_docs:
                shutil.copy(join(JSON_DOCS_FILEPATH, '{}.json'.format(doc)), test_path)
            # log
            if (test_nb + 1) % 10 == 0:
                print('Created {} tests'.format(test_nb+1))

    # Do document filtering
    f = open('resources/filtering_test/filtering_visualization.html', 'w+')
    for test_nb in range(NB_TESTS):
        test_path = join(TEST_BASE_PATH, 'test_{}'.format(test_nb+1))

        # Get doc2sents and doc2words
        doc2sents = read_json_docs(test_path)
        doc2words = {}
        for doc in doc2sents:
            words = flatten(doc2sents[doc])
            doc2words[doc] = [w[0] for w in words]

        filtered_docs, distracted_docs = docs_filtering(test_path)

        f.write('<h1>Test {}</h1>\n'.format(test_nb+1))
        f.write('<h3>Relevant Docs</h3>\n')
        for relevant_d in filtered_docs:
            context = ' '.join(doc2words[relevant_d][:150])
            f.write('<span style="color:blue">Document {}</span>: {} </br>\n'.format(relevant_d, context))
        f.write('<h3>Irrelevant Docs</h3>\n')
        for irrelevant_d in distracted_docs:
            context = ' '.join(doc2words[irrelevant_d][:150])
            f.write('<span style="color:red">Document {}</span>: {} </br>\n'.format(irrelevant_d, context))
        f.flush()
    f.close()
