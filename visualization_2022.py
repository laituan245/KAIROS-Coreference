import os
import copy
import json
import time
import torch
import random
import logging
import pywikibot

from os.path import join
from data import read_json_docs, locstr_to_loc
from argparse import ArgumentParser


def flatten(l):
    return [item for sublist in l for item in sublist]

# For Quizlet 4 docs
DOC2URL = {
    'L0C04AT6B': 'https://www.businessinsider.com/chipotle-closes-restaurant-after-illness-outbreak-2018-7',
    'L0C04AT6J': 'https://www.cnbc.com/2018/08/01/chipotle-hit-with-new-lawsuit-over-food-contamination-in-ohio.html',
    'L0C04AT6W': 'https://www.yahoo.com/lifestyle/hundreds-stricken-unknown-illness-apparently-162041489.html/',
    'L0C04AT6Y': 'https://www.justice.gov/opa/pr/chipotle-mexican-grill-agrees-pay-25-million-fine-and-enter-deferred-prosecution-agreement',
    'L0C04CJ13': '',
    'L0C04D4DA': 'https://abcnews.go.com/Health/Healthday/story?id=6646519&page=1',
    'L0C04D4DB': 'https://www.kcbd.com/story/9829807/timeline-of-events-in-salmonella-outbreak/',
    'L0C04D4DS': 'https://www.cdc.gov/salmonella/2009/peanut-butter-2008-2009.html',
    'L0C04D4RB': 'https://www.foodsafetynews.com/2018/09/former-food-safety-manager-for-pca-wants-supreme-court-to-review-her-case/'

}

def get_wikidata_title(wikidata_id):
    try:
        site = pywikibot.Site('wikidata', 'wikidata')
        repo = site.data_repository()
        item = pywikibot.ItemPage(repo, wikidata_id)
        item_dict = item.get() #Get the item dictionary
        return item_dict['labels']['en']
    except:
        return 'An error occured while retrieving wikidata title'

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum[0]

# Helper Function
def read_coref(coref_fp):
    cluster2mention = {}
    with open(coref_fp, 'r', encoding='utf-8') as f:
        for line in f:
            es = line.strip().split('\t')
            if not es[0] in cluster2mention:
                cluster2mention[es[0]] = {
                    'mentions': [],
                    'type': [],
                    'link': ['NA'],
                }
            if len(es) <= 4:
                if 'type' in es[1].lower():
                    cluster2mention[es[0]]['type'].append(es[2])
                if 'link' in es[1].lower():
                    cluster2mention[es[0]]['link'].append(es[2])
                continue
            if not 'mention' in es[1]: continue
            if not es[-2] in cluster2mention[es[0]]['mentions']:
                cluster2mention[es[0]]['mentions'].append(es[-2])
    return cluster2mention

def generate_visualization(docs, cluster2mention, output):
    with open(output, 'w+', encoding='utf-8') as f:
        for ix, c in enumerate(cluster2mention):
            # Type Info
            type_info = ''
            if len(cluster2mention[c]['type']) > 0:
                all_types = cluster2mention[c]['type']
                type_info = find_majority(all_types)
                type_info = f'Type = {type_info}'
            # Wikidata Linking Info
            wikidata_id = cluster2mention[c]['link'][-1]
            link_info = ''
            if wikidata_id != 'NA' and not (wikidata_id.startswith('NIL')):
                wikidata_title = get_wikidata_title(wikidata_id)
                link_info = f'<a href="https://www.wikidata.org/wiki/{wikidata_id}">{wikidata_id}</a>'
                link_info = f'Wikidata ID = {link_info} | Wikidata Title = {wikidata_title}'
            # Cluster Info (Combining Wikidata Info and Type Info)
            cluster_info = ''
            if len(type_info) > 0: cluster_info += type_info
            if len(link_info) > 0:
                if len(cluster_info) > 0: cluster_info += ' | '
                cluster_info += link_info
            if len(cluster_info) > 0:
                cluster_info = '(' + cluster_info + ')'

            # Write Cluster Line
            f.write('<hr><h3>Cluster {} {}</h3>'.format(ix+1, cluster_info))

            for m_index, m in enumerate(cluster2mention[c]['mentions']):
                doc_id, m_start, m_end = locstr_to_loc(m)
                if not doc_id in docs: continue
                tokens = flatten(docs[doc_id])
                token_texts = [t[0] for t in tokens]
                start_index, end_index = None, None
                for index, (_, start, end) in enumerate(tokens):
                    if m_start == start: start_index = index
                    if m_end == end: end_index = index
                assert(not start_index is None)
                assert(not end_index is None)
                mention_text = ' '.join(token_texts[start_index:end_index+1])
                left_context = ' '.join(token_texts[start_index-5: start_index])
                right_context = ' '.join(token_texts[end_index+1 : end_index+6])
                doc_info = doc_id
                if doc_id in DOC2URL:
                    doc_info = '<a href="{}">{}</a>'.format(DOC2URL[doc_id], doc_id)
                if len(cluster2mention[c]['type']) > 1:
                    f.write('Mention {} [Document {} ({})] {} <font color="red">{}</font> {} </br>'.format(m, doc_info, cluster2mention[c]['type'][m_index], left_context, mention_text, right_context))
                else:
                    f.write('Mention {} [Document {}] {} <font color="red">{}</font> {} </br>'.format(m, doc_info, left_context, mention_text, right_context))

def generate_visualization_for_cluster(docs, entity2mention, event2mention, cluster, cluster_nb, output_dir):
    docs = copy.deepcopy(docs)
    entity2mention = copy.deepcopy(entity2mention)
    event2mention = copy.deepcopy(event2mention)

    # Process docs ~ Remove document not in the cluster
    removed_ids = [k for k in docs if not k in cluster]
    for remove_id in removed_ids: del docs[remove_id]

    # Process entity2mention
    removed_entities = []
    for entity in entity2mention:
        m0 = list(entity2mention[entity]['mentions'])[0]
        m0 = m0.split(':')[0]
        if not m0 in cluster: removed_entities.append(entity)
    for entity in removed_entities: del entity2mention[entity]

    # Process event2mention
    removed_events = []
    for event in event2mention:
        m0 = list(event2mention[event]['mentions'])[0]
        m0 = m0.split(':')[0]
        if not m0 in cluster: removed_events.append(event)
    for event in removed_events: del event2mention[event]

    # Generate visualization files
    generate_visualization(docs, entity2mention, join(output_dir, 'cluster_{}_entity_coref.html'.format(cluster_nb)))
    generate_visualization(docs, event2mention, join(output_dir, 'cluster_{}_event_coref.html'.format(cluster_nb)))

def main(args_coref_dir, args_json_dir, args_output_dir):
    # Read json docs
    docs = read_json_docs(args_json_dir)
    print('Number of docs: {}'.format(len(docs)))

    # Read entity_coref and event_coref
    entity2mention = read_coref(join(args_coref_dir, 'entity.cs'))
    event2mention = read_coref(join(args_coref_dir, 'event.cs'))

    # Read doc clustering info
    clusters = []
    with open(join(args_coref_dir, 'clusters.txt'), 'r') as f:
        for line in f:
            clusters.append(json.loads(line))

    # generate_visualization_for_cluster
    for cluster_nb, cluster in enumerate(clusters):
        generate_visualization_for_cluster(docs, entity2mention, event2mention, cluster, cluster_nb, args_output_dir)

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--coref_dir', type=str, default='test/dryrun_bombing/all/')
    parser.add_argument('--json_dir', type=str, default='test/dryrun_bombing/json/')
    parser.add_argument('--output_dir', type=str, default='test/dryrun_bombing')
    args = parser.parse_args()

    main(args.coref_dir, args.json_dir, args.output_dir)
