# Language Models for Ancient Greek

Code for my B.Sc. thesis on Natural Language Processing (NLP) for the Ancient
Greek language. The paper can be found
[here](https://pergamos.lib.uoa.gr/uoa/dl/frontend/en/browse/3100154).

With the rise of the transformer Neural Network, Language Models (LMs) have been
created and trained for most known languages. The performance of those models
when fine-tuned on downstream tasks, surpasses all previous baselines by a
significant margin. Thus, it can be argued that pre-training LMs has now become
of paramount importance when aiming to achieve good performance on NLP tasks.

Surprisingly though, not many LMs have been created for Ancient Greek. After a
thorough research, only two models were found:

1. A character-level BERT by Brennan Nicholson, which can be found
[here](https://github.com/brennannicholson/ancient-greek-char-bert).
2. A token-level BERT by Pranaydeep et al., which can be found
[here](https://github.com/pranaydeeps/Ancient-Greek-BERT).

The former achieved decent results, but since it's character-level, it doesn't
have many use cases apart from Masked Language Modelling. The latter also
achieved good results when fine-tuned on the downstream task of Part-of-Speech
(PoS) Tagging.

The goal of the thesis was to create an improved Ancient Greek LM, by leveraging
new BERT-like architectures and pre-training methods. The results can be found
in the paper.
<br> </br>


# How to set up

It is strongly advised to use a Unix-like OS for this project, as most scripts
have been adapted to them.

1. Install [Anaconda](https://www.anaconda.com/) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Download the repo using the command
    ```shell
    git clone https://github.com/AndrewSpano/BSc-Thesis.git && cd BSc-Thesis
    ```

3. Create and activate a virtual environment using the command
    ```shell
    conda create --name ag-nlp-venv python=3.8 && conda activate ag-nlp-venv
    ```

4. Install the required packages using the command
    ```shell
    pip install -r requirements.txt
    ```
<br>


# Downloading the data

To download the data, clean it, preprocess it and train a BPE tokenizer, simply
run the script

```shell
python download_and_process_data.py
```

This script will

1. Download, clean the text data and save it in `data/plain-text/` (directory
will be overwritten if needed).
2. Train a tokenizer and save it in `objects/bpe_tokenizer/` (directory will be
overwritten if needed).
3. Use the tokenizer to create input IDs from the plain-text data and save it
in `data/processed-data/` (directory will be overwritten if needed).
4. Train a sklearn [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
on the PoS tags and save it in `objects/le.pkl` (overwriting it if needed).
<br> </br>


# Training

The [RoBERTa family of models](https://huggingface.co/docs/transformers/model_doc/roberta)
from the [transformers](https://github.com/huggingface/transformers) open-source
library has been used. For the implementation of the training process, two
different frameworks were used. Specifically,
[PyTorch Lightning](https://www.pytorchlightning.ai/) (PL) and the
[Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
from huggingface were used.
<br> </br>

## Pre-training with Masked Language Modelling

The RoBERTa model uses the auxiliary task of Masked Language Modelling
(MLM) to pre-train a Language Model.

### PyTorch Lightning

The PL model for MLM is implemented in the `LitRoBERTaMLM` class located in
[mlm_model.py](pl_models/mlm_model.py). To pre-train the LM with PL, the
script `pl_train_mlm.py` can be used. Its arguments can be found in the function
`parse_pl_mlm_input` located in [cmd_args.py](utils/cmd_args.py).
An example of running the script is:

```shell
python pl_train_mlm.py                                       \
    --logdir logs/pl-mlm/                                    \
    --config-path configurations/pl-mlm-example-config.ini   \
    --savedir objects/PL-AG-RoBERTa/                         \
    --plot-savepath plots/pl-mlm.png                         \
    --device cuda                                            \
    --distributed                                            \
    --seed random
```

### Huggingface Trainer

To pre-train the LM with the Trainer API from huggingface, the script
`hf_train_mlm.py` can be used. Its arguments can be found in the function
`parse_hf_mlm_input` located in [cmd_args.py](utils/cmd_args.py).
An example of running the script is:

```shell
python hf_train_mlm.py                                       \
    --logdir logs/hf-mlm/                                    \
    --config-path configurations/hf-mlm-example-config.ini   \
    --savedir objects/HF-AG-RoBERTa/                         \
    --plot-savepath plots/hf-mlm.png                         \
    --seed 3407
```
<br>


## Fine-tuning on Part-of-Speech Tagging

Once a LM has been pre-trained, its performance can be evaluated by
fine-tuning it to a downstream task and assessing the results on it. In this
repo, the downstream task that was chosen is Part-of-Speech (PoS) Tagging.

### PyTorch Lightning

The PL model for PoS Tagging is implemented in the class `PoSRoBERTa` which
is located in [pos_model.py](pl_models/pos_model.py). To fine-tune the LM
with PL, the script `pl_train_pos.py` can be used. Its arguments can be found
in the function `parse_pl_pos_input` located in
[cmd_args.py](utils/cmd_args.py). An example of running the script is:

```shell
python pl_train_pos.py                                       \
    --logdir logs/pl-pos/                                    \
    --config-path configurations/pl-pos-example-config.ini   \
    --pre-trained-model objects/PL-AG-RoBERTa/               \
    --savedir objects/PL-PoS-AG-RoBERTa                      \
    --plot-savepath plots/pl-pos.png                         \
    --confusion-matrix plots/pl-pos-cm.png                   \
    --device cuda                                            \
    --distributed
```

### Huggingface Trainer

To fine-tune the LM on PoS Tagging with the Trainer API from huggingface, the
script [hf_train_pos.py](hf_train_pos.py) can be used. Its arguments can be
found in the function `parse_hf_pos_input` located in
[cmd_args.py](utils/cmd_args.py). An example of running the script is:

```shell
python hf_train_pos.py                                       \
    --logdir logs/hf-pos/                                    \
    --config-path configurations/hf-pos-example-config.ini   \
    --pre-trained-model objects/HF-AG-RoBERTa/               \
    --savedir objects/HF-PoS-AG-RoBERTa                      \
    --plot-savepath plots/hf-pos.png                         \
    --confusion-matrix plots/hf-pos-cm.png                   \
    --seed 3
```
<br>


# Hyperparameter Tuning

Hyperparameter tuning in order to minimize the MLM validation loss during
pre-training has also been implemented for both frameworks. Specifically, the
[HyperOpt](https://github.com/hyperopt/hyperopt) library was used. The search
spaces can be found in the files [pl_tune_mlm.py](pl_tune_mlm.py) and
[hf_tune_mlm.py](hf_tune_mlm.py). The only command line argument that they
accept is the number of maximum evaluations to perform, which by default is 100
if it's not provided. An example of running the scripts is:

1. PyTorch Lightning
   ```shell
   python pl_tune_mlm.py --max-evals 150
   ```

2. Huggingface
   ```shell
   python hf_tune_mlm.py
   ```
