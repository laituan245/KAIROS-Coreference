import os
import time
import torch

from constants import *
from argparse import ArgumentParser
from utils import load_tokenizer_and_model, get_predicted_antecedents, flatten, create_dir_if_not_exist
from data import EntityCentricDocument, EntityCentricDocumentPair, load_entity_centric_dataset
from algorithms import UndirectedGraph
from os.path import join, dirname

def get_cluster_labels(clusters, id2mention, field):
    clusterlabels, nil_ctx = [], 0
    for c in clusters:
        count = {}
        for mid in c:
            m = id2mention[mid]
            if field == 'fb_id' and field in m and m[field].startswith('NIL'): continue
            if field in m: count[m[field]] = count.get(m[field], 0) + 1
        if len(count) == 0:
            label = 'no_label_{}'.format(str(nil_ctx))
            nil_ctx += 1
        else:
            label = max(count, key=lambda k: count[k])
        clusterlabels.append(label)
    return clusterlabels

def propagate(start_id, mid2linkid, adjList):
    visited = set()
    stack = list(adjList[start_id])
    visited.add(start_id)
    while len(stack) > 0:
        curr = stack.pop()
        if curr in visited: continue
        if not curr in mid2linkid:
            mid2linkid[curr] = mid2linkid[start_id]
            visited.add(curr)
            stack.extend(adjList[curr])
    return visited


def entity_coref(cs_path, json_dir, fb_linking_path, output_path, language, filtered_doc_ids, clusters, english_docs, spanish_docs,
                 predicted_pairs = set(), mid2linkid = {}):
    create_dir_if_not_exist(dirname(output_path))

    # Read the original entity.cs
    mid2lines, oe2mid, cur_type = {}, {}, None
    mid2mentiontype = {}
    with open(cs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            es = line.split('\t')
            if len(line) == 0: continue
            if len(es) <= 4:
                if es[1].strip().lower() == 'type':
                    cur_type = es[2].strip()
                continue
            mid = es[-2]
            oe2mid[es[0]] = mid
            if not mid in mid2lines: mid2lines[mid] = []
            if not es[1].startswith('canonical_mention'):
                mid2lines[mid].append(es)
                mid2mentiontype[mid] = es[1]

    # Read the original relation.cs file (in the same directory as entity.cs)
    relation_cs = join(dirname(cs_path), 'relation.cs')
    relation_pairs = set()
    with open(relation_cs, 'r', encoding='utf-8') as f:
        for line in f:
            es = line.strip().split('\t')
            arg0, arg1 = oe2mid[es[0]], oe2mid[es[2]]
            relation_pairs.add((arg0, arg1))
            relation_pairs.add((arg1, arg0))

    # Load tokenizer and model
    if language == 'en': tokenizer, model = load_tokenizer_and_model(EN_ENTITY_MODEL)
    elif language == 'es' or language == 'cross': tokenizer, model = load_tokenizer_and_model(ES_ENTITY_MODEL)

    # Load dataset
    print('Loading dataset')
    entities, docs = load_entity_centric_dataset(tokenizer, cs_path, json_dir, fb_linking_path, filtered_doc_ids)
    mentions = flatten([e['mentions'].values() for e in entities.values()])

    # Build mid2type
    mid2type = {}
    for e in entities.values():
        e_type = e['type']
        for m in e['mentions']:
            mid2type[m] = e_type

    # Build id2mention
    id2mention, fb2mentions = {}, {}
    for m in mentions:
        id2mention[m['mention_id']] = m
        if 'fb_id' in m:
            if not m['fb_id'] in fb2mentions: fb2mentions[m['fb_id']] = []
            fb2mentions[m['fb_id']].append(m)

    # Build doc2cluster
    doc2cluster = {}
    for ix, c in enumerate(clusters):
        for doc_id in c:
            doc2cluster[doc_id] = ix

    # Initialize mid2linkid
    for m in mentions:
        if 'fb_id' in m and not m['fb_id'].startswith('NIL'):
            mid2linkid[m['mention_id']] = m['fb_id']

    # Apply the coref model
    start_time = time.time()
    if True:
        adjList = {}
        for (node1, node2) in predicted_pairs:
            if not node1 in adjList: adjList[node1] = set()
            adjList[node1].add(node2)
            if not node2 in adjList: adjList[node2] = set()
            adjList[node2].add(node1)
        with torch.no_grad():
            # Main loop
            for z in range(2):
                for i in range(len(docs)):
                    doci = docs[i]
                    if z == 0:
                        # Within-doc coref
                        start_range = len(docs)
                        end_range = len(docs)+1
                    else:
                        # Cross-doc coref
                        start_range = i+1
                        end_range = len(docs)
                    for j in range(start_range, end_range):
                        if j == len(docs):
                            # Dummy doc
                            docj = EntityCentricDocument(doci.doc_id, [], [], None)
                        else:
                            docj = docs[j]
                        if language == 'cross':
                            if doci.doc_id in english_docs and docj.doc_id in english_docs: continue
                            if doci.doc_id in spanish_docs and docj.doc_id in spanish_docs: continue
                        if len(doci.words) == 0 and len(docj.words) == 0: continue
                        if doc2cluster[doci.doc_id] != doc2cluster[docj.doc_id]: continue
                        inst = EntityCentricDocumentPair(doci, docj, tokenizer)
                        doc_entities = inst.entity_mentions

                        preds = model(*inst.tensorized_example)[1]
                        preds = [x.cpu().data.numpy() for x in preds]
                        mention_starts, mention_ends, top_antecedents, top_antecedent_scores = preds
                        predicted_antecedents, predicted_scores = get_predicted_antecedents(top_antecedents, top_antecedent_scores, True)

                        # Extract predicted pairs
                        for ix, (s, e) in enumerate(zip(mention_starts, mention_ends)):
                            cur_predicted_score = predicted_scores[ix]
                            if predicted_antecedents[ix] >= 0:
                                antecedent_idx = predicted_antecedents[ix]
                                mention_1 = doc_entities[ix]
                                mention_2 = doc_entities[antecedent_idx]
                                # Extract fields
                                m1_id = mention_1['mention_id']
                                m2_id = mention_2['mention_id']
                                m1_text = mention_1['canonical_mention']
                                m2_text = mention_2['canonical_mention']
                                type_1, mention_type_1 = mid2type[mention_1['mention_id']], mid2mentiontype[m1_id]
                                type_2, mention_type_2 = mid2type[mention_2['mention_id']], mid2mentiontype[m2_id]
                                # Rule: Two mentions that are linked to two different entities are not coreferential
                                if 'fb_id' in mention_1 and 'fb_id' in mention_2:
                                    link1 = mention_1['fb_id']
                                    link2 = mention_2['fb_id']
                                    if (not link1.startswith('NIL')) and (not link2.startswith('NIL')) \
                                    and link1 != link2:
                                        continue
                                # Rule: Not linking two mentions from two different documents if one of them is a pronoun
                                if mention_1['doc_id'] != mention_2['doc_id']:
                                    if mention_type_1 == 'pronominal_mention' or mention_type_2 == 'pronominal_mention':
                                        continue
                                # Rule: Not link a NIL mention to a non-NIL mention if the two are from two different documents.
                                if mention_1['doc_id'] != mention_2['doc_id']:
                                    link1 = mention_1['fb_id']
                                    link2 = mention_2['fb_id']
                                    if link1.startswith('NIL') and not link2.startswith('NIL'):
                                        continue
                                    if link2.startswith('NIL') and not link1.startswith('NIL'):
                                        continue
                                # Rule: Not link two people' mentions if their names do not match.
                                if 'PER' in type_1 and 'PER' in type_2 and mention_type_1 == 'mention' and mention_type_2 == 'mention':
                                    m1_text_words = set(m1_text.lower().split())
                                    m2_text_words = set(m2_text.lower().split())
                                    should_skip = True
                                    if len(m1_text_words.intersection(m2_text_words)) == min(len(m1_text_words), len(m2_text_words)):
                                        should_skip = False
                                    if should_skip: continue
                                # Rule: Don't apply coref if the two mentions are from the same document and we are doing cross-doc coref
                                if doci.doc_id != docj.doc_id and mention_1['doc_id'] == mention_2['doc_id']:
                                    continue
                                # Rule: Check mid2linkid
                                if m1_id in mid2linkid and m2_id in mid2linkid:
                                    link1 = mid2linkid[m1_id]
                                    link2 = mid2linkid[m2_id]
                                    if link1 != link2:
                                        continue
                                # Append the pair to predicted_pairs
                                predicted_pairs.add((m1_id, m2_id))
                                if not m1_id in adjList: adjList[m1_id] = set()
                                if not m2_id in adjList: adjList[m2_id] = set()
                                adjList[m1_id].add(m2_id)
                                adjList[m2_id].add(m1_id)
                                # Update mid2linkid
                                if m1_id in mid2linkid and not m2_id in mid2linkid:
                                    mid2linkid[m2_id] = mid2linkid[m1_id]
                                    propagate(m2_id, mid2linkid, adjList)
                                if m2_id in mid2linkid and not m1_id in mid2linkid:
                                    mid2linkid[m1_id] = mid2linkid[m2_id]
                                    propagate(m1_id, mid2linkid, adjList)

        f.close()
    print("--- Applying the entity coref model took %s seconds ---" % (time.time() - start_time))

    # Build clusters
    graph = UndirectedGraph([m['mention_id'] for m in mentions])
    print('Number of vertices: {}'.format(graph.V))

    # If two mentions are linked to the same entity, then they are coreferential
    for ix in range(len(mentions)):
        for jx in range(ix+1, len(mentions)):
            if 'fb_id' in mentions[ix] and 'fb_id' in mentions[jx]:
                mi_fb_id = mentions[ix]['fb_id']
                mj_fb_id = mentions[jx]['fb_id']
                if mi_fb_id == mj_fb_id and (not mi_fb_id.startswith('NIL')) and (not mj_fb_id.startswith('NIL')):
                    predicted_pairs.add((mentions[ix]['mention_id'], mentions[jx]['mention_id']))

    # Add edges from predicted_pairs
    filtered_predicted_pairs = set()
    for node1, node2 in predicted_pairs:
        if (node1, node2) in relation_pairs or (node2, node1) in relation_pairs: continue
        if mid2type[node1] != mid2type[node2]: continue
        if node1 in mid2linkid and node2 in mid2linkid:
            if mid2linkid[node1] != mid2linkid[node2]:
                continue
        # Add edges
        filtered_predicted_pairs.add((node1, node2))
        graph.addEdge(node1, node2)
    predicted_pairs = filtered_predicted_pairs
    # Get connected components (with-in doc clusters)
    print('Get connected components')
    clusters = sccs = graph.getSCCs()
    assert(len(flatten(sccs)) == graph.V)

    # Read the fb_linking_path to get canonical_mention and mention
    in_fb_linking = set()
    with open(fb_linking_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            es = line.split('\t')
            if len(line) == 0: continue
            if len(es) <= 3: continue
            mid = es[-2]
            if not mid in in_fb_linking:
                in_fb_linking.add(mid)
                mid2lines[mid] = []
            mid2lines[mid].append(es)

    # Outputs
    nil_count = 0
    prefix = ':Entity_EDL_'
    clusters.sort(key=lambda x: len(x), reverse=True)
    c_types = get_cluster_labels(clusters, id2mention, field='type')
    c_links = get_cluster_labels(clusters, id2mention, field='fb_id')
    with open(output_path, 'w+', encoding='utf-8') as f:
        for ix, (type, link, cluster) in enumerate(zip(c_types, c_links, clusters)):
            ix_str = str(ix)
            while len(ix_str) < 7: ix_str = '0' + ix_str
            es_0 = prefix + ix_str
            # Type
            type_line = '\t'.join([es_0, 'type', type])
            f.write('{}\n'.format(type_line))
            # Element in clusters
            for m in cluster:
                for line in mid2lines[m]:
                    line[0] = es_0
                    mention_line = '\t'.join(line)
                    f.write('{}\n'.format(mention_line))
            # Link
            if not link.startswith('no_label_'):
                link_line = '\t'.join([es_0, 'link', link])
                f.write('{}\n'.format(link_line))
            else:
                nil_count += 1
                link = str(nil_count)
                while len(link) < 9: link = '0' + link
                link = 'NIL' + link
                link_line = '\t'.join([es_0, 'link', link])
                f.write('{}\n'.format(link_line))


    model.to(torch.device('cpu'))

    # Delete unused objects
    del(model)
    del(tokenizer)

    return predicted_pairs, mid2linkid
