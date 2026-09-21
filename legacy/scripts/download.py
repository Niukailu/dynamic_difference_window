''' pull all experiments '''
import argparse
import logging
from aligo import Aligo


def parse_options():
    ''' parse options '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='file name')
    args = parser.parse_args()
    return args.name


def download(experiment):
    ''' download '''
    ali = Aligo(level=logging.ERROR)
    file = ali.get_folder_by_path(f'experiments/{experiment}')
    ali.download_folder(file.file_id, local_folder='experiments')


if __name__ == '__main__':
    download(parse_options())
