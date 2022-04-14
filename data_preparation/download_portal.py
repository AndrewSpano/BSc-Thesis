import os
import re
import random
import requests

from pathlib import Path
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin


from data_preparation.data_prep_utils import (
    clean_raw_text,
    get_sentences,
    print_stats_and_save
)
from data_preparation.download_all import MLM_TARGET_DIR


def preprocess(text: str) -> str:
    """Applies a series of preprocessing steps, specific to the portal
        website."""
    # remove brackets with text inside them, eg: [61.3]
    text = re.sub(r'\[[a-zA-Z0-9.]+]', '', text)

    # remove parenthesis with text inside them, eg: (61.3)
    text = re.sub(r'\([a-zA-Z0-9.]+\)', '', text)

    # fixes words have been split in two
    text = re.sub(r'-\n', '', text)

    # remove some symbols because for some reason they exist randomly in words
    text = re.sub(r'[\[\]…»]', '', text)

    # convert multiple dots to one
    text = re.sub(r'\.+', '.', text)

    # substitutes multiple white spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # removes leading author/speaker names
    delimiters = r'[\.;;!]\s'
    text = '. '.join(
        [sentence.strip() for sentence in re.split(delimiters, text) if
         len(sentence.strip().split()) > 1 or not sentence.strip().isupper()]
    )

    return text


def download_portal(dest_dir: Path):
    train_sentences, val_sentences, test_sentences = [], [], []

    random.seed(80085)

    root_url = 'https://www.greek-language.gr/greekLang/ancient_greek/tools/' \
               'corpora/translation/contents.html'
    html = requests.get(root_url).text
    soup = BeautifulSoup(html, 'lxml')
    results = soup.find('div', {'class': 'height'}).find('ul').findAll('li')

    for result in tqdm(results, desc='Downloading Portal data'):

        author_href = result.find('a')['href']
        author_url = urljoin(root_url, author_href)
        author_html = requests.get(author_url).text

        soup2 = BeautifulSoup(author_html, 'lxml')
        author_stuff = soup2.find('div', {'class': 'height'})

        # skip first ul because it contains anchors to other uls
        for idx, ul in enumerate(author_stuff.findAll('ul')[1:]):
            for li in ul.findAll('li'):

                a = li.find('a')
                text_href = a['href']
                text_url = urljoin(author_url, text_href)
                text_html = requests.get(text_url).text
                soup3 = BeautifulSoup(text_html, 'lxml')

                # skip first 2 because they are useless
                text_sentences = []
                for paragraph in soup3.findAll('p')[2:]:
                    clean_text = clean_raw_text(preprocess(paragraph.text))
                    paragraph_sentences = get_sentences(clean_text)
                    text_sentences += paragraph_sentences

                # 90 - 5 - 5 split
                prob = random.uniform(0, 1)
                if prob < 0.9:
                    train_sentences.append(text_sentences)
                elif prob < 0.95:
                    val_sentences.append(text_sentences)
                else:
                    test_sentences.append(text_sentences)

    # save the MLM data
    print_stats_and_save(train_sentences, val_sentences, test_sentences,
                         dest_dir=dest_dir, name='portal')


if __name__ == "__main__":

    os.makedirs(MLM_TARGET_DIR/'train', exist_ok=True)
    os.makedirs(MLM_TARGET_DIR/'val', exist_ok=True)
    os.makedirs(MLM_TARGET_DIR/'test', exist_ok=True)

    download_portal(MLM_TARGET_DIR)
