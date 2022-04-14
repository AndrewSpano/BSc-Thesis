"""
Inspired by

https://github.com/brennannicholson/ancient-greek-char-bert/blob/master/data_prep/greek_data_prep/clean_data.py
"""

import os
import re
import random
import shutil

from pathlib import Path
from tqdm.auto import tqdm
from bs4 import BeautifulSoup

from data_preparation.data_prep_utils import (
    get_files,
    clean_texts,
    download_git_repo,
    split_texts,
    print_stats_and_save
)
from data_preparation.download_all import MLM_TARGET_DIR

# exclude Bacchylides' Odes due to the fragmentary nature of the text
BACHCHYLIDES_ODES = [
    'tlg0199.tlg001.perseus-grc1.xml',
    'tlg0199.tlg002.perseus-grc1.xml',
]

TEXTS_WITH_SIGNIFICANT_OCR_ERRORS = [
    'tlg3129.ogl001.1st1K-grc1.xml',  # In Cyrilli In XII Prophetas Theophylacti
]

# ideally these should be automatically identified and converted
# but CLTK's parser doesn't seem to work...
BETA_CODE_FILES = [
    'tlg2003.tlg002.perseus-grc1.xml',
    'tlg2003.tlg010.perseus-grc1.xml',
    'tlg2003.tlg007.perseus-grc1.xml',
    'tlg2003.tlg009.perseus-grc1.xml',
    'tlg2003.tlg004.perseus-grc1.xml',
    'tlg2003.tlg012.perseus-grc1.xml',
    'tlg2003.tlg008.perseus-grc1.xml',
    'tlg2003.tlg005.perseus-grc1.xml',
    'tlg2003.tlg011.perseus-grc1.xml',
    'tlg2003.tlg001.perseus-grc1.xml',
    'tlg2003.tlg003.perseus-grc1.xml',
    'tlg2003.tlg006.perseus-grc1.xml',
]

FILES_CAUSING_PARSING_ERRORS = [
    'tlg2003.tlg013.perseus-grc1.xml',
    'tlg2003.tlg017.perseus-grc1.xml ',
    'tlg2040.tlg002.perseus-grc1.xml',
    'tlg2040.tlg004.perseus-grc1.xml',
    'tlg0648.tlg001.perseus-grc1.xml',
    'tlg2018.tlg002.perseus-grc1.xml',
    'tlg0363.tlg007.perseus-grc1.xml',
    'tlg0058.tlg001.perseus-grc1.xml',
    'tlg2003.tlg017.perseus-grc1.xml',
    'tlg0099.tlg001.perseus-grc1.xml',
    'tlg0556.tlg001.perseus-grc1.xml',
    'tlg0019.tlg007.perseus-grc1.xml',
    'tlg0019.tlg007.perseus-grc1.xml',
    'tlg0284.tlg029.perseus-grc1.xml',
    'tlg0284.tlg026.perseus-grc1.xml',
    'tlg0284.tlg046.perseus-grc1.xml',
    'tlg0284.tlg045.perseus-grc1.xml',
    'tlg0284.tlg048.perseus-grc1.xml',
    'tlg0284.tlg054.perseus-grc1.xml',
    'tlg0284.tlg009.perseus-grc1.xml',
    'tlg0284.tlg004.perseus-grc1.xml',
    'tlg0284.tlg035.perseus-grc1.xml',
    'tlg0284.tlg022.perseus-grc1.xml',
    'tlg0641.tlg001.perseus-grc1.xml',
]
PERSEUS_FILES_TO_EXCLUDE = (
    BACHCHYLIDES_ODES
    + BETA_CODE_FILES
    + FILES_CAUSING_PARSING_ERRORS
    + TEXTS_WITH_SIGNIFICANT_OCR_ERRORS
)


def parse_xml(fp):
    """Parses a Perseus XML file. The approach here is very rough. Ideally the
        nodes should be traversed and those containing a significant amount of
        non-greek characters removed
        (see: https://github.com/ThomasK81/TEItoCEX)"""
    soup = BeautifulSoup(fp, 'xml')
    # remote 'note' tags
    for note_tag in soup.find_all('note'):
        note_tag.decompose()
    assert len(list(soup.find_all('note'))) == 0
    # get all remaining text
    raw_text = soup.find('text').get_text()
    return raw_text


def parse_perseus_xml(files):
    """Parses the given XML files. Returns the parsed files as well as a list
        of files which couldn't be parsed at all."""
    raw_texts = []
    failed_to_parse = []
    for filename in tqdm(files, desc='Parsing Perseus Data'):
        with open(filename) as fp:
            raw_text = parse_xml(fp)
        if not raw_text:
            failed_to_parse.append(filename)
        else:
            raw_texts.append(raw_text)
    return raw_texts, failed_to_parse


def download_perseus(dest_dir: Path):

    # download the repo
    perseus_repo = 'https://github.com/PerseusDL/canonical-greekLit'
    download_git_repo(perseus_repo, 'master')
    perseus_data_dir = Path('canonical-greekLit')/'data'

    # get the files with ancient greek text from the repo
    perseus_regex = re.compile(r'grc[0-9]*\.xml$')
    perseus_files = get_files(perseus_data_dir, perseus_regex,
                              PERSEUS_FILES_TO_EXCLUDE)

    # parse them
    perseus_raw_texts, _ = parse_perseus_xml(perseus_files)
    # clean them, as much as possible
    perseus_data = clean_texts(perseus_raw_texts)

    # apply the standard cleaning and assign it to a dataset
    random.seed(80085)
    data = split_texts(perseus_data, train_fraction=0.95)

    # save the MLM data
    print_stats_and_save(*data, dest_dir=dest_dir, name='perseus')

    # remove the repo
    shutil.rmtree(perseus_data_dir.parent)


if __name__ == "__main__":

    os.makedirs(MLM_TARGET_DIR/'train', exist_ok=True)
    os.makedirs(MLM_TARGET_DIR/'val', exist_ok=True)
    os.makedirs(MLM_TARGET_DIR/'test', exist_ok=True)

    download_perseus(MLM_TARGET_DIR)
