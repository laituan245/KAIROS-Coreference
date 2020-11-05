from coref import coref_main
from utils import read_cluster_info

ONEIE_OUTPUT = '/shared/nas/data/m1/yinglin8/kairos/result/kairos_all_oct_27/m1_m2'
LINKING_OUTPUT = '/shared/nas/data/m1/xiaoman6/tmp/20200920_kairos_linking/output/kairos_all_oct_27/m1_m2/linking/en.linking.wikidata.cs'
COREFERENCE_OUTPUT = '/shared/nas/data/m1/tuanml2/tmp/fast_coref/'

if __name__ == "__main__":
    clusters = read_cluster_info('resources/processed/clusters.txt')
    coref_main(ONEIE_OUTPUT, LINKING_OUTPUT, COREFERENCE_OUTPUT, clusters)
