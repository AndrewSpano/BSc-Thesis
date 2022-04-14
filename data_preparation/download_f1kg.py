"""
Inspired by

https://github.com/brennannicholson/ancient-greek-char-bert/blob/master/data_prep/greek_data_prep/clean_data.py
"""

import os
import re
import glob
import shutil
import random

from pathlib import Path

from data_preparation.data_prep_utils import (
    download_and_unzip,
    clean_texts,
    get_files,
    split_texts,
    print_stats_and_save
)
from data_preparation.download_all import MLM_TARGET_DIR


def get_f1kg_texts(files):
    """Gets the specified F1KG text files (which do not need parsing,
        unlike the Perseus files)."""
    texts = []
    for i, f in enumerate(files):
        with open(f, 'r') as fp:
            texts.append(fp.read())
    return texts


def download_f1kg(dest_dir: Path) -> None:

    url = 'https://zenodo.org/record/2592513/files/OpenGreekAndLatin/' \
          'First1KGreek-1.1.4529.zip'
    temp_dest = Path('temp-f1kg')
    download_and_unzip(url, temp_dest)

    # get the data files
    f1kg_regex = re.compile(r'grc[0-9]*\.txt$')
    f1kg_dir = glob.glob(str(temp_dest/'OpenGreekAndLatin*'/'text'))[0]
    f1kg_files = get_files(f1kg_dir, f1kg_regex, [])

    # parse them and clean them as much as possible
    f1kg_raw_texts = get_f1kg_texts(f1kg_files)
    f1kg_data = clean_texts(f1kg_raw_texts)

    # apply the standard cleaning and assign it to a dataset
    random.seed(80085)
    data = split_texts(f1kg_data, train_fraction=0.95)

    # save the MLM data
    print_stats_and_save(*data, dest_dir=dest_dir, name='f1kg')

    # remove the temporary directory with the downloaded data
    shutil.rmtree(temp_dest)


if __name__ == "__main__":

    os.makedirs(MLM_TARGET_DIR/'train', exist_ok=True)
    os.makedirs(MLM_TARGET_DIR/'val', exist_ok=True)
    os.makedirs(MLM_TARGET_DIR/'test', exist_ok=True)

    download_f1kg(MLM_TARGET_DIR)
