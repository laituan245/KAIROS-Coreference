import os
import json
import time
import torch
import random
import logging

from utils import flatten
from data import read_json_docs, locstr_to_loc
from argparse import ArgumentParser

# For Quizlet 4 docs
DOC2URL = {
    'K0C0448WM': 'https://kutv.com/news/local/utah-trooper-hit-from-behind-by-runaway-car-in-sardine-canyon',
    'K0C0448WL': 'https://kutv.com/news/local/police-find-missing#:~:text=POTUS%2FFLOTUS%20quarantining-,Two%20bodies%20found%20in%20mine%20near%20Eureka,to%20be%20missing%20Utah%20teens&text=EUREKA%2C%20Utah%20%E2%80%94%20(KUTV),Breezy%20Otteson%20and%20Riley%20Powell.',
    'K0C0448WJ': 'https://www.theguardian.com/world/2018/aug/14/venezuela-crisis-maduro-raise-fuel-prices-combat-smuggling',
    'K0C0448WI': 'https://apnews.com/article/b7c41edb7d5042c8a60e3bfffcc1ecff#:~:text=OAS%20Secretary%20General%20Luis%20Almagro,a%20region%2Dwide%20migration%20crisis.',
    'K0C047Z59': 'https://www.stgeorgeutah.com/news/archive/2019/04/24/mgk-teen-in-pine-view-bomb-scare-is-sentenced-to-probation/#:~:text=George%20News-,ST.,%E2%80%9Cintensely%20supervised%E2%80%9D%20probation%20Wednesday.',
    'K0C047Z57': 'https://bnonews.com/index.php/2018/03/utah-teen-arrested-after-bringing-bomb-to-school/',
    'K0C041NHY': 'https://www.newsbreak.com/news/1356225947561/teen-who-brought-explosives-to-pine-view-high-sentenced-to-jail-time-probation?s=oldSite&ss=website',
    'K0C041NHW': 'https://tripwire.dhs.gov/news/209313',
    'K0C041NHV': 'https://www.abc4.com/news/juvenile-charged-with-bringing-bomb-to-school/',
    'K0C047Z5A': 'https://www.vox.com/world/2018/8/5/17653002/venezuela-drone-attack-nicolas-maduro',
    'K0C041O37': 'https://www.yahoo.com/news/m/b24450e7-e24e-3f68-885a-e302d0598894/analysis-backs-claim-drones.html',
    'K0C041O3D': 'https://www.wetalkuav.com/dji-responds-to-drone-attacks/'
}

# Helper Function
def read_coref(coref_fp):
    cluster2mention = {}
    with open(coref_fp, 'r') as f:
        for line in f:
            es = line.strip().split('\t')
            if not es[0] in cluster2mention:
                cluster2mention[es[0]] = {
                    'mentions': set()
                }
            if len(es) <= 4:
                if 'type' in es[1].lower():
                    cluster2mention[es[0]]['type'] = es[2]
                continue
            if not 'mention' in es[1]: continue
            cluster2mention[es[0]]['mentions'].add(es[-2])
    return cluster2mention

def generate_visualization(docs, cluster2mention, output):
    with open(output, 'w+') as f:
        for ix, c in enumerate(cluster2mention):
            type_info = cluster2mention[c]['type']
            f.write('<hr><h3>Cluster {} (Type = {})</h3>'.format(ix+1, type_info))
            for m in cluster2mention[c]['mentions']:
                doc_id, m_start, m_end = locstr_to_loc(m)
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
                f.write('[Document {}] {} <font color="red">{}</font> {} </br>'.format(doc_info, left_context, mention_text, right_context))

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--json_dir', default='resources/quizlet4/output/oneie/m1_m2/json')
    parser.add_argument('--entity_coref', default='resources/quizlet4/output/coref/entity.cs')
    parser.add_argument('--event_coref', default='resources/quizlet4/output/coref/event.cs')
    args = parser.parse_args()

    # Read json docs
    docs = read_json_docs(args.json_dir)
    print('Number of docs: {}'.format(len(docs)))

    # Read entity_coref and event_coref
    entity2mention = read_coref(args.entity_coref)
    event2mention = read_coref(args.event_coref)

    # Generate visualization files
    generate_visualization(docs, entity2mention, 'resources/quizlet4/output/coref/entity_coref.html')
    generate_visualization(docs, event2mention, 'resources/quizlet4/output/coref/event_coref.html')
