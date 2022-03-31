import json

thing_contaminated_role = 'artifactexistence.contamination.unspecified_ThingContaminated.actual'
destination_role = 'Movement.Transportation.Unspecified_Destination.actual'

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
            if not es[0] in event2arguments:
                event2arguments[es[0]] = []
            event2arguments[es[0]].append(line)

    # Check if the same entity is playing multiple roles:
    filtered_lines = set()
    for eid in event2arguments:
        arguments = event2arguments[eid]
        arguments = [l.split('\t') for l in arguments]
        entity2roles = {}
        for argument in arguments:
            if not argument[2] in entity2roles:
                entity2roles[argument[2]] = []
            entity2roles[argument[2]].append(argument[1])
        for e in entity2roles:
            entity2roles[e] = list(set(entity2roles[e]))
            if len(entity2roles[e]) > 1:
                if entity2type[e] == 'ORG' and thing_contaminated_role in entity2roles[e]:
                    entity2roles[e].remove(thing_contaminated_role)
                if entity2type[e] == 'PER' and destination_role in entity2roles[e]:
                    entity2roles[e].remove(destination_role)
        for argument in event2arguments[eid]:
            es_argument = argument.split('\t')
            entity = es_argument[2]
            role = es_argument[1]
            if not role in entity2roles[entity]:
                filtered_lines.add(argument)


    # Output
    with open(event_cs, 'w+') as f:
        for line in output_lines:
            if line in filtered_lines: continue
            f.write(line)
