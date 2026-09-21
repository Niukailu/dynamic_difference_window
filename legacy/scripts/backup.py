''' backup '''
import os
import sys
import logging
import argparse
import tqdm
from aligo import Aligo
try:
    from utils import zip
except Exception:
    sys.path.append(sys.path[0].replace('/scripts', ''))
    from utils import zip


def parse_options():
    ''' parse options '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='file name')
    args = parser.parse_args()
    return args.name


def zip_folders(name):
    ''' zip all sub floders '''
    floders = [f'experiments/{name}/{floder}' for floder in os.listdir(f'experiments/{name}/')]
    floders = [floder for floder in floders if os.path.isdir(floder)]
    for floder in tqdm.tqdm(floders, desc='压缩'):
        zip(floder)


def upload(name):
    ''' upload top floder '''
    ali = Aligo(name='falcon', level=logging.ERROR)
    ali.upload_folder(f'experiments/{name}', parent_file_id=ali.get_folder_by_path("experiments/").file_id)


def main():
    ''' main '''
    name = parse_options()
    zip_folders(name)
    upload(name)


if __name__ == '__main__':
    main()
