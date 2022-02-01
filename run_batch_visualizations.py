import argparse
import os
import visualization_2022

from os import listdir
from os.path import isdir, join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coref_base_dir',
                        help='Base directory (coref). Each subdir is a complex event',
                        default='/shared/nas/data/m1/tuanml2/kairos_dryrun2')
    parser.add_argument('--oneie_base_dir',
                        help='Base directory (oneie). Each subdir is a complex event',
                        default='/shared/nas/data/m1/pengfei4/tools/event_weak_supervision/kairos_dryrun2/')
    parser.add_argument('--base_output_dir',
                        default='coref_visualizations')
    args = parser.parse_args()

    subdirs = [dn for dn in listdir(args.coref_base_dir) if isdir(join(args.coref_base_dir, dn))]
    if not os.path.exists(args.base_output_dir):
        os.makedirs(args.base_output_dir)

    for subdir in subdirs:
        coref_dir = join(args.coref_base_dir, subdir, 'all')
        json_dir = join(args.oneie_base_dir, subdir, 'json')
        output_dir = join(args.base_output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        visualization_2022.main(coref_dir, json_dir, output_dir)

