''' backup '''
import os
import sys
import logging
from aligo import Aligo


def main():
    ''' main '''
    name = sys.argv[1]
    ali = Aligo(name='falcon', level=logging.ERROR)
    ali.upload_folder(f'experiments/{name}', parent_file_id=ali.get_file_by_path("experiments/").file_id)

if __name__ == '__main__':
    main()
