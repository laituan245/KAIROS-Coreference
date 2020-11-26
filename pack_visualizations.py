import shutil
from os.path import join
from utils import create_dir_if_not_exist

TEST_NBS = [1, 2]
OUTPUT_PATH = 'caci_visualizations'

create_dir_if_not_exist(OUTPUT_PATH)
for test_nb in TEST_NBS:
    test_output_path = join(OUTPUT_PATH, 'ce100{}'.format(test_nb))
    create_dir_if_not_exist(test_output_path)
    input_path = 'test/caci/all{}'.format(test_nb)
    visualizations = join(input_path, 'visualizations')
    visualizations_ordered = join(input_path, 'visualizations_ordered')
    cluster_fp = join(input_path, 'clusters.txt')
    with open(cluster_fp, 'r') as f:
        lines = []
        for line in f:
            if len(line.strip()) == 0: continue
            lines.append(line.strip())
    nb_clusters = len(lines)
    for ix in range(nb_clusters):
        if ix == 0:
            destination_path = join(test_output_path, 'main G')
        else:
            destination_path = join(test_output_path, 'distractor {}'.format(ix))
        create_dir_if_not_exist(destination_path)
        shutil.copy(join(visualizations, 'cluster_{}_entity_coref.html'.format(ix)), destination_path)
        shutil.copy(join(visualizations_ordered, 'cluster_{}_event_coref.html'.format(ix)), destination_path)
