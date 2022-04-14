import os
import re
import pickle
import shutil
import random
import subprocess
import unicodedata

from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple, Union
from cltk.corpus.utils.formatter import cltk_normalize


CHARS_TO_REMOVE = """{}΄|Ϛݲ§ϛ♃5ᾱᾅὝᾂ̆ᾦ#Ἦ*ᾆ⟩ὋἎὒὮ′̣ϝὯἾ͵ῂüὬ⌋⌈‚•ä+̀ö&""" \
                  """–ͅᾕë1͂ῲᾡἇἛἋGϠ¶%ῢ^ἊἯæᾇ\\2ᾁῡἚ̋/⌉Ὢß!⌊Ἣ=ῗóΌ`3ï⌞⌟ᾲΆ65ϡ̈4∗Ὣq""" \
                  """═òΈϞ○à7áΊᾒ‖~אϟΪϥ›\u200b⁄‹íé⋖ὊÍ9Ἲ̄8{ῧ}Üᾟᾍᾨ―ΉŕὟ⩹✶0Ώᾯᾥᾌ""" \
                  """\x8d⟦⟧\x9a¦ᾬἻ£a⋇ῐ¬ÓbῚŒἿἏÉῌᾃ\x98°ΎῈÁ⨆�""" \
                  """ç↑ạὛ⏔⏑̅✝ú\x9dᾺụᾢᾓᾘῼùÒSῠϙ─\x90לṕᾣô\x9cῸᾜḿ$⦵⊏ī\x9eֹ\x8e""" \
                  """ÌćÆ\x8fǁ⊻@ū÷Ҁ∾ῺìῪ\x8c\x81ᾮᾈèÿœῩῊ\x88⊤З♀⊙\xadÄÖᾞߤ⁑⸨""" \
                  """\x8aḍ⫯∼źẂ⋆★Ῑᾩ‵ᾎý√⏝⏓⏕ṃ×ȳហḾti¿⥽⥼⊣⊔ӄẉ͎\u070eҏďĎ̠◻ᾰ""" \
                  """\ue1c2rƑ̧\x7fេឲិតាỳӕῘLᾙΫẃ☾☿♂♄⊢⋃Ā±TMĹ€║̇čō"""


CHARS_TO_REPLACE = {
    '∠': 'Δ',
    '△': 'Δ',
    '\ufeff': ' ',  # ZERO WIDTH NO-BREAK SPACE (U+FEFF)
    'ῑ': 'ϊ',
    '✝': '†',
    # regularize angled brackets
    '<': '⟨',
    '>': '⟩',
    '〈': '⟨',
    '〉': '⟩',
    # regularize quotation marks
    '“': '"',
    '”': '"',
    '«': '"',
    '»': '"',
    '„': '"',
    '‟': '"',
    '‹': "'",
    '›': "'",
    '‘': "'",
    '':  "'",
}


def clean_raw_text(text: str) -> str:
    """Cleans a piece of raw text and returns it."""
    def normalize(str_: str) -> str:
        """Normalizes a piece of text with the usage of decomposition by
            canonical equivalence, which leads to stripping accents."""
        non_accent_characters = [
            char for char in unicodedata.normalize('NFD', str_)
            if unicodedata.category(char) != 'Mn'
        ]
        return ''.join(non_accent_characters)
    # lower surprisingly also works perfectly for unicode characters
    return normalize(text).lower()


def remove_braces(text: str) -> str:
    """Removes various braces and brackets from the text."""
    # remove square and curly braces and their contents (if less than 40 chars)
    text = re.sub(r'{[^}]{,40}}', '', text)
    # remove text between ⟦ and ⟧ as the pairs were manually checked,
    # there is no length restriction
    text = re.sub(r'⟦[^⟧]*⟧', '', text)
    return text


def remove_unwanted_chars(token: str) -> str:
    """Removes all characters in the list from the token."""
    output = filter(lambda c: c not in CHARS_TO_REMOVE, token)
    return ''.join(output)


def replace_chars(token: str) -> str:
    """Replaces all characters in the dict with their specified replacement."""
    for c in token:
        if c in CHARS_TO_REPLACE:
            token = re.sub(c, CHARS_TO_REPLACE[c], token)
    return token


def clean_tokens(tokens: List[str]) -> List[str]:
    """Cleans a list of tokens."""
    cleaned_tokens = []
    for token in tokens:
        if token:
            # remove words in which latin characters appear
            if not re.search(r'\w+', token, re.ASCII):
                # remove tokens containing digits
                if not re.search(r'\d+', token):
                    # normalize
                    token = cltk_normalize(token)
                    token = token.strip('\t\r\n')
                    # remove unwanted chars
                    token = remove_unwanted_chars(token)
                    token = replace_chars(token)
                    # remove any inter-word hyphens or en-dashes
                    token = re.sub(r'([^\s])([-–])', r'\1', token)
                    cleaned_tokens.append(token)
    return cleaned_tokens


def clean_texts(raw_texts: List[str]) -> List[List[str]]:
    """Cleans a list of texts and splits it in documents."""
    data = []
    for i, raw_text in enumerate(tqdm(raw_texts, desc='Cleaning texts')):
        text = ''
        raw_text = remove_braces(raw_text).splitlines()
        for line in raw_text:
            if line != '\n':
                tokens = line.split(' ')
                cleaned_tokens = clean_tokens(tokens)
                tokens = ' '.join([t for t in cleaned_tokens])
                text += tokens
            else:
                text += '\n'
        # remove multiple dots
        text = re.sub(r'(\.*\s*\.)+', '.', text)
        # remove empty parentheses
        text = re.sub(r'\(\s*\)', '', text)
        # remove empty angled brackets
        text = re.sub(r'⟨\s*⟩', '', text)
        # remove other noise
        text = re.sub(r'[\[\]()«]', '', text)
        # split the text in documents
        documents = re.split(r'[.;!]\n{2,}', text)
        for document in documents:
            # clean it, get the sentences of the document and add them
            document_text = clean_raw_text(re.sub(r'\s+', ' ', document))
            document_sentences = get_sentences(document_text)
            data.append(document_sentences)
    return data


def get_files(
        directory: Union[Path, str],
        regex: re.Pattern,
        files_to_exclude: List[str]
) -> List[str]:
    """Finds files matching the regex in the specified directory except
        for those in the exclusion list. Returns a list of file paths."""
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if re.search(regex, f) and f not in files_to_exclude:
                files.append(dirpath + '/' + f)
    return files


def get_sentences(text: str) -> List[str]:
    """Split a corpus of Ancient Greek text into sentences."""
    # returns a list of the form:
    # ['sentence-1', 'delimiter-1', 'sentence-2', 'delimiter-2', ...]
    delimiters_pattern = r'([\.;;!])'
    sentences_and_delimiters = re.split(delimiters_pattern, text)
    # concatenate the delimiters to the sentences and return them
    sentences = [
        (sentences_and_delimiters[i - 1] +
         sentences_and_delimiters[i]).strip()
        for i in range(1, len(sentences_and_delimiters), 2)
    ]
    # remove sentences with two characters or fewer
    sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 2]
    return sentences


def write_sentences(filepath: Path, sentences: List[List[str]]) -> None:
    """Writes the given sentences in the specified path, separating them by
        a newline character."""
    with open(filepath, 'w') as fp:
        fp.write('\n\n'.join('\n'.join(sentence) for sentence in sentences))


def save_pickle(savepath: Path, data: object) -> None:
    """Saves the given data in pickle format the specified location."""
    with open(savepath, 'wb') as fp:
        pickle.dump(data, fp)


def download_and_unzip(address: str, dest_folder: Path) -> None:
    """Downloads and unzips the file from the given address."""
    if os.path.isdir(dest_folder):
        shutil.rmtree(dest_folder)
    filename = address.split('/')[-1]
    subprocess.run(['wget', '-O', filename, address])
    subprocess.run(['unzip', filename, '-d', dest_folder])
    os.unlink(filename)


def download_git_repo(address: str, commit: str) -> None:
    """Clones the given repo and checks out to the specified branch."""
    repo_name = address.split("/")[-1]
    if os.path.isdir(repo_name):
        shutil.rmtree(repo_name)
    subprocess.run(['git', 'clone', address])
    subprocess.run(['git', 'checkout', commit], cwd=repo_name)


def split_texts(texts: List[List[str]], train_fraction: float) -> \
        Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """Cleans each text in a list of texts, gets the sentences and splits
        them to train/val/test according to the train_fraction argument."""
    train_sentences, val_sentences, test_sentences = [], [], []
    for document_sentences in texts:
        # train_fraction, (1 - train_fraction)/2,  (1 - train_fraction)/2  split
        prob = random.uniform(0, 1)
        if prob < train_fraction:
            train_sentences.append(document_sentences)
        elif prob < train_fraction + (1 - train_fraction) / 2:
            val_sentences.append(document_sentences)
        else:
            test_sentences.append(document_sentences)

    return train_sentences, val_sentences, test_sentences


def print_stats_and_save(
        train_sentences: List[List[str]],
        val_sentences: List[List[str]],
        test_sentences: List[List[str]],
        dest_dir: Path,
        name: str
) -> None:
    """Prints statistics regarding the data and then saves it."""
    print(f'{name} - Number of train sentences: '
          f'{sum(map(len, train_sentences))}')
    print(f'{name} - Number of val sentences: '
          f'{sum(map(len, val_sentences))}')
    print(f'{name} - Number of test sentences: '
          f'{sum(map(len, test_sentences))}')

    write_sentences(dest_dir/'train'/f'{name}-train.txt', train_sentences)
    write_sentences(dest_dir/'val'/f'{name}-val.txt', val_sentences)
    write_sentences(dest_dir/'test'/f'{name}-test.txt', test_sentences)
