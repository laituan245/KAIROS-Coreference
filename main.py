from coref import coref_main
from utils import read_cluster_info

ONEIE_OUTPUT = '/shared/nas/data/m1/tuanml2/oneie_with_relation'
LINKING_OUTPUT = '/shared/nas/data/m1/xiaoman6/tmp/20200920_kairos_linking/output/oneie_with_relation/linking/en.linking.wikidata.cs'
COREFERENCE_OUTPUT = '/shared/nas/data/m1/tuanml2/tmp/coref/'

if __name__ == "__main__":
    clusters = read_cluster_info('resources/original/clusters.txt')
    coref_main(ONEIE_OUTPUT, LINKING_OUTPUT, COREFERENCE_OUTPUT, clusters)
