import argparse
from os import listdir
from os.path import isdir, join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help='Base directory. Each subdir is a complex event',
                        required=True)
    args = parser.parse_args()

    subdirs = [dn for dn in listdir(args.base_dir) if isdir(join(args.base_dir, dn))]
