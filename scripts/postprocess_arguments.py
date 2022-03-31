import json

def read_cs(fp):
    lines = []
    with open(fp, 'r') as f:
        for line in f:
            lines.append(line)
    return lines

def postprocess_arguments(entity_cs, event_cs, ontology_fp):
    # Read cs files
    entity_lines = read_cs(entity_cs)
    event_lines = read_cs(event_cs)
    with open(ontology_fp, 'r') as f:
        ontology = json.loads(f.read())
    _ontology = {}
    for k, v in ontology.items():
        _ontology[k.lower()] = v
    ontology = _ontology

    # Build entity2type
    entity2type = {}
    for l in entity_lines:
        es = l.strip().split('\t')
        if es[1] == 'type':
            entity2type[es[0]] = es[2]

    # Process events file
    event2type = {}
    event2arguments, output_lines = {}, []
    for line in event_lines:
        es = line.strip().split('\t')
        if es[1] in ['type', 'modality'] or es[1].startswith('mention') \
        or es[1].startswith('canonical_mention'):
            if es[1] == 'type':
                event2type[es[0]] = es[2]
            output_lines.append(line)
            continue
        event_type = event2type[es[0]]
        roles = ontology[event_type.lower()]['roles']
        cur_role = es[1][es[1].rfind('_')+1:es[1].rfind('.')]
        if cur_role in roles:
            output_lines.append(line)

    # Output
    with open(event_cs, 'w+') as f:
        for line in output_lines:
            f.write(line)
