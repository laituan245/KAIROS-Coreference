def add_types_qnode(args_inpath):
    type2qnode = {
        'FAC': 'Q13226383',
        'ORG': 'Q43229',
        'WEA': 'Q728',
        'GPE': 'Q27096213',
        'LOC': 'Q2221906',
        'VEH': 'Q42889',
        'PER': 'Q215627'
    }
    type2label = {
        'FAC': 'Facility',
        'ORG': 'Organization',
        'WEA': 'Weapon',
        'GPE': 'Geopolitical entity',
        'LOC': 'Location',
        'VEH': 'Vehicle',
        'PER': 'Person'
    }
    eid2type = {}
    all_lines = []
    with open(args_inpath, 'r') as f:
        for line in f:
            all_lines.append(line)
            es = line.strip().split('\t')
            if len(es) < 2: continue
            if es[1] == 'type':
                cur_type = es[-1].split()[-1]
                eid2type[es[0]] = cur_type

    with open(args_inpath, 'w+') as f:
        for line in all_lines:
            es = line.strip().split('\t')
            if len(es) > 2 and es[1] == 'link' and es[-1].startswith('NIL'):
                try:
                    qnode = type2qnode[eid2type[es[0]]]
                    qlabel = type2label[eid2type[es[0]]]
                except:
                    qnode = 'Q35120'
                    qlabel = 'Entity'
                # Write qnode
                es[-1] = qnode
                f.write('\t'.join(es) + '\n')
                # Write label
                new_es = [es[0], 'qlabel', qlabel]
                new_line = '\t'.join(new_es)
                f.write('{}\n'.format(new_line))
            else:
                f.write(line)
