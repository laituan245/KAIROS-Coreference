from os.path import join
from collections import Counter
from utils import read_event_types

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def read_entity(entity_cs):
    entity2type, mid2entity, entity2mention = {}, {}, {}
    with open(entity_cs, 'r', encoding='utf-8') as f:
        for line in f:
            es = line.strip().split('\t')
            if len(es) <= 3:
                if es[1] == 'type': entity2type[es[0]] = es[-1]
                continue
            if 'mention' in es[1]:
                mid2entity[es[3]] = es[0]
            if 'canonical_mention' in es[1]:
                entity2mention[es[0]] = es[2][1:-1]
    return entity2type, mid2entity, entity2mention

def read_event(event_cs, event_types):
    all_lines = []
    event2type, event2args, event2mention = {}, {}, {}
    with open(event_cs, 'r', encoding='utf-8') as f:
        for line in f:
            all_lines.append(line)
            es = line.strip().split('\t')
            if len(es) <= 3:
                if es[1] == 'type':
                    event2type[es[0]] = es[2]
                continue
            if not (es[1].startswith('mention') or es[1].startswith('canonical_mention')):
                event_type = event2type[es[0]]
                if event_type in event_types: # Consider only events in the KAIROS ontology
                    event_args = event_types[event_type]['args']
                    arg_name = es[1].split('.')[-2].split('_')[-1]
                    if not arg_name in event_args: continue
                    arg_nb = event_args[arg_name]
                    if not es[0] in event2args: event2args[es[0]] = {}
                    if not arg_nb in event2args[es[0]]: event2args[es[0]][arg_nb] = []
                    event2args[es[0]][arg_nb].append((es[-3].strip(), line))
            elif 'canonical_mention' in es[1]:
                event2mention[es[0]] = es[2][1:-1]
    return all_lines, event2type, event2args, event2mention

def remove_arguments(output_entity_cs, output_event_cs, output_path):
    event_types = read_event_types('resources/event_types.tsv')
    entity2type, mid2entity, entity2mention = read_entity(output_entity_cs)
    all_event_lines, event2type, event2args, event2mention = read_event(output_event_cs, event_types)

    #
    removed_lines = set()

    # Clean up arg1/arg2
    for e in event2args:
        cur_args = event2args[e]
        if (not 'attack' in event2type[e].lower()): continue
        for arg_name in ['<arg1>', '<arg2>']:
            if arg_name in cur_args:
                args1 = cur_args[arg_name]
                # Build proper_nouns
                has_major_proper_noun = False
                proper_nouns = [entity2mention[a[0]] for a in args1 if a[0] in entity2mention]
                proper_nouns = [p for p in proper_nouns if p[0].isupper()]
                if find_majority(proper_nouns)[1] > 1:
                    has_major_proper_noun = True
                if has_major_proper_noun:
                    for a in args1:
                        if a[0] in entity2mention:
                            mention = entity2mention[a[0]]
                            if mention[0].islower(): removed_lines.add(a[1])


    # arg1 and arg2 must not be the same
    for e in event2args:
        cur_args = event2args[e]
        if '<arg1>' in cur_args and '<arg2>' in cur_args:
            counter1 = Counter([a[0] for a in cur_args['<arg1>']])
            counter2 = Counter([a[0] for a in cur_args['<arg2>']])
            for first_arg in cur_args['<arg1>']:
                for second_arg in cur_args['<arg2>']:
                    if first_arg[0] == second_arg[0]:
                        ctx1 = counter1[first_arg[0]]
                        ctx2 = counter2[second_arg[0]]
                        if ctx1 > ctx2:
                            removed_lines.add(second_arg[1])
                        elif ctx1 < ctx2:
                            removed_lines.add(first_arg[1])

    # New Output
    with open(output_event_cs, 'w+', encoding='utf-8') as f:
        for line in all_event_lines:
            if line in removed_lines: continue
            f.write('{}'.format(line))

    # Log
    log_output_path = join(output_path, 'remove_args_logs.html')
    with open(log_output_path, 'w+') as log_f:
        for remove_line in removed_lines:
            es = remove_line.strip().split('\t')
            event_mention = event2mention[es[0]]
            arg_name = es[1]
            entity_mention = entity2mention[es[2]]
            print('In event <b>{}</b>, remove <span style="color:red;">{}</span> as <span style="color:blue;">{}</span>'.format(event_mention, entity_mention, arg_name))
