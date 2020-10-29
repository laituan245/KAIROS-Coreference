wrong_file = '/shared/nas/data/m1/tuanml2/kairos/cross_lingual_coref/entity.cs'
correct_file = '/shared/nas/data/m1/tuanml2/kairos/cross_lingual_coref_fixedname/entity.cs'
output_file = '/shared/nas/data/m1/tuanml2/kairos/cross_lingual_coref/tmp_entity.cs'

def read_cs(cs_path):
    lines, mid2entity, entity2canonical = [], {}, {}
    with open(cs_path, 'r') as f:
        for line in f:
            lines.append(line.strip())
            es = line.strip().split('\t')
            if len(es) <= 3:
                continue
            if es[1].startswith('canonical_mention'):
                entity2canonical[es[0]] = es[2][1:-1].strip()
                mid = es[-2].strip()
                mid2entity[mid] = es[0]
    return lines, mid2entity, entity2canonical

old_lines, old_mid2entity, old_entity2canonical = read_cs(wrong_file)
new_lines, new_mid2entity, new_entity2canonical = read_cs(correct_file)

with open(output_file, 'w+') as f:
    for line in old_lines:
        es = line.strip().split('\t')
        if es[1].startswith('canonical_mention'):
            mid = es[-2].strip()
            entity = new_mid2entity[mid]
            canonical_mention = new_entity2canonical[entity]
            es[2] = '"{}"'.format(canonical_mention)
            line = '\t'.join(es)
            f.write('{}\n'.format(line))
        else:
            f.write('{}\n'.format(line))
