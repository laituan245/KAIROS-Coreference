from scripts.es_retriever import ESCandidateRetriever

def add_qlabel(entity_cs_fp):
    '''
    Input:
        entity_cs_fp: Path to the entity cs file (with qnodes already added)
    '''
    cg = ESCandidateRetriever()
    # Get all the unique qnodes
    all_input_lines = []
    qnode_ids = set()
    with open(entity_cs_fp, 'r') as f:
        for line in f:
            all_input_lines.append(line)
            line = line.strip()
            if len(line) == 0: continue
            es = line.split('\t')
            if es[1] == 'link':
                if not es[2].startswith('NIL'):
                    qnode_ids.add(es[2])
    print(f'Number of unique qnode ids: {len(qnode_ids)}')
    # Query ES
    datas = cg.search_entities_by_ids(list(qnode_ids))
    wikibase2label = {}
    for data in datas:
        wikibase = data['data']['wikibase']
        label = data['data']['label']
        wikibase2label[wikibase] = label
    # Output
    with open(entity_cs_fp, 'w+') as f:
        for line in all_input_lines:
            f.write(line)
            es = line.strip().split('\t')
            if len(es) > 1 and es[1] == 'link':
                if not es[2].startswith('NIL'):
                    qlabel = wikibase2label[es[2].strip()]
                    new_es = [es[0], 'qlabel', qlabel]
                    new_line = '\t'.join(new_es)
                    f.write('{}\n'.format(new_line))
