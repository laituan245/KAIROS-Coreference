from os.path import join
from scripts.es_retriever import ESCandidateRetriever

cg = ESCandidateRetriever()

for _ce in ['ce2013', 'ce2020', 'ce2079', 'ce2103']:
    entity_fp = join('resources', 'dryrun2023', _ce, 'en', 'entity.cs')
    with open(entity_fp, 'r', encoding='utf8') as f:
        lines = f.readlines()

    with open(entity_fp, 'w+', encoding='utf8') as f:
        for line in lines:
            f.write(line)
            es = line.strip().split('\t')
            if es[1] == 'link':
                datas = cg.search_entities_by_ids([es[-1]])
                try:
                    desc = datas[0]['data']['description']
                except:
                    desc = datas[0]['data']['label']
                es[1] = 'wd_description'
                es[2] = desc
                data_l = '\t'.join(es).strip()
                f.write(data_l)
                f.write('\n')
