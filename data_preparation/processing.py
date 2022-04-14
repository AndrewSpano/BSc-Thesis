import os
import pickle
import re
import shutil

from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from sklearn.preprocessing import LabelEncoder

from data_preparation.download_all import MLM_TARGET_DIR, POS_TARGET_DIR


TOKENIZER_PATH = Path(__file__).parent.parent/'objects'/'bpe_tokenizer'
LABEL_ENCODER_PATH = Path(__file__).parent.parent/'objects'/'le.pkl'
PROCESSED_DATA_PATH = Path(__file__).parent.parent/'data'/'processed-data'


def train_and_save_tokenizer(vocab_size: int = 30522, min_frequency: int = 2):
    # ensure the tokenizer directory is empty
    if os.path.isdir(TOKENIZER_PATH):
        shutil.rmtree(TOKENIZER_PATH)
    os.makedirs(TOKENIZER_PATH)

    # get the train text
    train_files = sorted(list(map(str, (MLM_TARGET_DIR/'train').glob('*'))))

    # create, train a tokenizer on the train text data and save it
    byte_level_bpe_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    special_tokens = [
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '<mask>'
    ]
    byte_level_bpe_tokenizer.train(
        files=train_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens
    )
    byte_level_bpe_tokenizer.save_model(str(TOKENIZER_PATH))
    byte_level_bpe_tokenizer.save(str(TOKENIZER_PATH/'config.json'))


def convert_data_to_mlm_format(max_length: int = 512):
    # ensure the processed MLM data directory is empty
    if os.path.isdir(PROCESSED_DATA_PATH/'MLM'):
        shutil.rmtree(PROCESSED_DATA_PATH/'MLM')
    os.makedirs(PROCESSED_DATA_PATH/'MLM')

    # load the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)

    def encode_sentences_of_file(filepath: Path) -> List[List[int]]:
        # read the text of the file
        with open(filepath, 'r') as fp:
            contents = fp.read()

        # documents are separated by two newlines when saved
        documents = re.split(r'\n{2,}', contents)

        # add chunks of data of maximum length 512 (including bos/eos tokens)
        data = []
        current = [tokenizer.bos_token_id]

        # encode the sentences of each document
        loop = tqdm(documents, desc=f'Converting {filepath} to input IDs')
        for document in loop:
            # sentences are separated by a newline when saved
            sentences = document.split('\n')

            # for each sentence, add it to the current segment if it fits,
            #  else end the current segment and create a new one with it
            for sentence in sentences:
                # skip bos/eos tokens as they are added manually
                encoded = tokenizer.encode(sentence, max_length=max_length,
                                           truncation=True)[1:-1]
                if len(current) + len(encoded) + 1 > max_length:
                    data.append(current + [tokenizer.eos_token_id])
                    current = [tokenizer.bos_token_id] + encoded
                else:
                    current += encoded

            # create a new segment from the last sentences of the document
            if len(current) > 1:
                data.append(current + [tokenizer.eos_token_id])

        return data

    # get all the files
    train_files = (MLM_TARGET_DIR/'train').glob('*')
    val_files = (MLM_TARGET_DIR/'val').glob('*')
    test_files = (MLM_TARGET_DIR/'test').glob('*')

    def transform_and_save_texts(files: List[Path], prefix: str) -> None:
        data = []
        for file in files:
            data += encode_sentences_of_file(file)
        with open(PROCESSED_DATA_PATH/'MLM'/f'{prefix}-data.pkl', 'wb') as fp:
            pickle.dump(data, fp)

    # transform and save the train/val/test texts to input IDs
    transform_and_save_texts(train_files, 'train')
    transform_and_save_texts(val_files, 'val')
    transform_and_save_texts(test_files, 'test')


def convert_data_to_pos_format():
    # ensure the processed PoS data directory is empty
    if os.path.isdir(PROCESSED_DATA_PATH/'PoS'):
        shutil.rmtree(PROCESSED_DATA_PATH/'PoS')
    os.makedirs(PROCESSED_DATA_PATH/'PoS')

    # load the label encoder and the tokenizer
    le = LabelEncoder()
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)

    def create_pos_data(
            sentences: List[List[str]],
            pos_tags: List[List[str]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        # lists to be returned
        input_ids, labels = [], []

        # iterate over all the sentences of the PoS Dataset
        loop = tqdm(zip(sentences, pos_tags), total=len(sentences),
                    desc='Preprocessing')
        for sentence, tags in loop:

            # initialize the input ID, label for the current sentence
            #  note that Cross-Entropy-Loss by default ignores index -100
            sentence_input_ids = [tokenizer.bos_token_id]
            sentence_labels = [-100]

            # convert each token of the sentence to its ID and get the label(s)
            for token, tag in zip(sentence, tags):
                ids = tokenizer(token)['input_ids'][1:-1]
                tag = le.transform([tag])[0]
                sentence_input_ids += ids
                # if a word is split in many sub-words, keep the original label
                #  only for the last sub-word, and use -100 for the others
                if len(ids) > 1:
                    sentence_labels += [-100] * (len(ids) - 1)
                sentence_labels.append(tag)

            # truncate long sentences
            sentence_input_ids = sentence_input_ids[:511]
            sentence_labels = sentence_labels[:511]

            # add bos token
            sentence_input_ids.append(tokenizer.eos_token_id)
            sentence_labels.append(-100)

            # add to final lists
            input_ids.append(sentence_input_ids)
            labels.append(sentence_labels)

        return input_ids, labels

    def load_pos_data(prefix: str) -> Tuple[List[str], List[str]]:
        root_dir = POS_TARGET_DIR/prefix
        with open(root_dir/f'diorisis-{prefix}-sentences.pkl', 'rb') as fp_:
            sentences = pickle.load(fp_)
        sentences = [sentence for doc in sentences for sentence in doc]
        with open(root_dir/f'diorisis-{prefix}-labels.pkl', 'rb') as fp_:
            pos_tags = pickle.load(fp_)
        pos_tags = [label for doc in pos_tags for label in doc]
        return sentences, pos_tags

    def save_pos_data(data: Tuple[List[str], List[str]], prefix: str) -> None:
        input_ids, labels = create_pos_data(*data)
        root_dir = PROCESSED_DATA_PATH/'PoS'
        with open(root_dir/f'pos-{prefix}-input-ids.pkl', 'wb') as fp_:
            pickle.dump(input_ids, fp_)
        with open(root_dir/f'pos-{prefix}-labels.pkl', 'wb') as fp_:
            pickle.dump(labels, fp_)
        print(f'Successfully saved the {prefix} data.')

    # --------------------- train data --------------------- #
    train_data = load_pos_data(prefix='train')

    # fit label encoder with the PoS tags
    all_classes = list(set([cls for tags in train_data[1] for cls in tags]))
    le.fit(sorted(all_classes))

    save_pos_data(train_data, prefix='train')

    # --------------------- validation data --------------------- #
    val_data = load_pos_data(prefix='val')
    save_pos_data(val_data, prefix='val')

    # --------------------- test data --------------------- #
    test_data = load_pos_data(prefix='test')
    save_pos_data(test_data, prefix='test')

    # save the label encoder
    with open(LABEL_ENCODER_PATH, 'wb') as fp:
        pickle.dump(le, fp)


def process_data():
    train_and_save_tokenizer()
    convert_data_to_mlm_format()
    convert_data_to_pos_format()


if __name__ == "__main__":
    process_data()
