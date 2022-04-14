import argparse

from pathlib import Path
from typing import List, Optional


def parse_pl_mlm_input(arg: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses the input for the PyTorch Lightning MLM training script."""
    default_logdir = Path(__file__).parent.parent/'logs'/'pl-mlm'
    default_savedir = Path(__file__).parent.parent/'objects'/'PL-AG-RoBERTa'
    default_seed = 'random'

    parser = argparse.ArgumentParser(description='PL MLM Training script.')

    # logs directory
    parser.add_argument('-l', '--logdir', type=Path, action='store',
                        metavar='logs-directory', default=default_logdir,
                        help='Path to the tensorboard logs directory.')
    # optional config file
    parser.add_argument('-c', '--config-path', type=Path, action='store',
                        metavar='configuration-file',
                        help='Path to the configuration file that will be '
                             'used to set the hyperparameters of the model.')
    # path to save the directory where the model will be saved
    parser.add_argument('-s', '--savedir', type=Path, action='store',
                        default=default_savedir, metavar='model-save-directory',
                        help='Path to the directory where the pre-trained '
                             'model will be saved.')
    # optional path to save plot with learning curves
    parser.add_argument('-p', '--plot-savepath', type=Path, action='store',
                        metavar='path-to-save-learning-curves-plot',
                        help='Path to the a .png filename where the learning '
                             'curves for the current experiment will be saved.')
    # device to use
    parser.add_argument('-d', '--device', type=str, action='store',
                        metavar='train-device', choices=['auto', 'cpu', 'cuda'],
                        default='auto', help='Which device to train on.')
    # whether to use multiple GPUs (in 1 node)
    parser.add_argument('--distributed', action='store_true',
                        help='Whether to train in a distributed fashion using '
                             'many GPUs (across 1 node). CUDA must be '
                             'available for this option.')
    # random seed
    parser.add_argument('--seed', type=str, action='store',
                        default=default_seed, metavar='seed',
                        help='Random seed to be set. Used for reproducibility '
                             'purposes. If not specified, then a random seed '
                             'will be set.')

    return parser.parse_args(arg)


def parse_pl_pos_input(arg: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses the input for the PyTorch Lightning PoS Tagging training
        script."""
    default_logdir = Path(__file__).parent.parent/'logs'/'pl-pos'
    default_loaddir = Path(__file__).parent.parent/'objects'/'PL-AG-RoBERTa'
    default_savedir = Path(__file__).parent.parent/'objects'/'PL-PoS-AG-RoBERTa'
    default_seed = 'random'

    desc = 'PL PoS Tagging Training script.'
    parser = argparse.ArgumentParser(description=desc)

    # logs directory
    parser.add_argument('-l', '--logdir', type=Path, action='store',
                        metavar='logs-directory', default=default_logdir,
                        help='Path to the tensorboard logs directory.')
    # optional config file
    parser.add_argument('-c', '--config-path', type=Path, action='store',
                        metavar='configuration-file',
                        help='Path to the configuration file that will be '
                             'used to set the hyperparameters of the model.')
    # path to load the model from
    parser.add_argument('-t', '--pre-trained-model', type=Path, action='store',
                        metavar='pre-trained-mlm', default=default_loaddir,
                        help='The path to the pre-trained MLM Ancient Greek '
                             'RoBERTa model.')
    # path to save the directory where the model will be saved
    parser.add_argument('-s', '--savedir', type=Path, action='store',
                        default=default_savedir, metavar='model-save-directory',
                        help='Path to the directory where the PoS tagging '
                             'fine-tuned model will be saved.')
    # optional path to save plot with learning curves
    parser.add_argument('-p', '--plot-savepath', type=Path, action='store',
                        metavar='path-to-save-learning-curves-plot',
                        help='Path to the .png filename where the learning '
                             'curves for the current experiment will be saved.')
    # optional path to save the confusion matrix of the test set
    parser.add_argument('-m', '--confusion-matrix', type=Path, action='store',
                        metavar='path-to-confusion-matrix-heatmap',
                        help='Path to the .png filename where the confusion '
                             'matrix of the test set will be saved.')
    # device to use
    parser.add_argument('-d', '--device', type=str, action='store',
                        metavar='train-device', choices=['auto', 'cpu', 'cuda'],
                        default='auto', help='Which device to train on.')
    # whether to use multiple GPUs (in 1 node)
    parser.add_argument('--distributed', action='store_true',
                        help='Whether to train in a distributed fashion using '
                             'many GPUs (across 1 node). CUDA must be '
                             'available for this option.')
    # random seed
    parser.add_argument('--seed', type=str, action='store',
                        default=default_seed, metavar='seed',
                        help='Random seed to be set. Used for reproducibility '
                             'purposes. If not specified, then a random seed '
                             'will be set.')

    return parser.parse_args(arg)


def parse_hf_mlm_input(arg: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses the input for the Hugging Face MLM training script."""
    default_logdir = Path(__file__).parent.parent/'logs'/'hf-mlm'
    default_savedir = Path(__file__).parent.parent/'objects'/'HF-AG-RoBERTa'
    default_seed = 'random'

    parser = argparse.ArgumentParser(description='HF MLM Training script.')

    # logs directory
    parser.add_argument('-l', '--logdir', type=Path, action='store',
                        metavar='logs-directory', default=default_logdir,
                        help='Path to the tensorboard logs directory.')
    # optional config file
    parser.add_argument('-c', '--config-path', type=Path, action='store',
                        metavar='configuration-file',
                        help='Path to the configuration file that will be '
                             'used to set the hyperparameters of the model.')
    # path to save the directory where the model will be saved
    parser.add_argument('-s', '--savedir', type=Path, action='store',
                        default=default_savedir, metavar='model-save-directory',
                        help='Path to the directory where the pre-trained '
                             'model will be saved.')
    # optional path to save plot with learning curves
    parser.add_argument('-p', '--plot-savepath', type=Path, action='store',
                        metavar='path-to-save-learning-curves-plot',
                        help='Path to the a .png filename where the learning '
                             'curves for the current experiment will be saved.')
    # whether to omit cuda
    parser.add_argument('-n', '--no-cuda', action='store_true',
                        help='If toggled, then no GPUs will be used for '
                             'training. Else, all available GPUs will be used.')
    # random seed
    parser.add_argument('--seed', type=str, action='store',
                        default=default_seed, metavar='seed',
                        help='Random seed to be set. Used for reproducibility '
                             'purposes. If not specified, then a random seed '
                             'will be set.')

    return parser.parse_args(arg)


def parse_hf_pos_input(arg: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses the input for the Hugging Face PoS Tagging training script."""
    default_logdir = Path(__file__).parent.parent/'logs'/'hf-pos'
    default_loaddir = Path(__file__).parent.parent/'objects'/'HF-AG-RoBERTa'
    default_savedir = Path(__file__).parent.parent/'objects'/'HF-PoS-AG-RoBERTa'
    default_seed = 'random'

    desc = 'HF PoS Tagging Training script.'
    parser = argparse.ArgumentParser(description=desc)

    # logs directory
    parser.add_argument('-l', '--logdir', type=Path, action='store',
                        metavar='logs-directory', default=default_logdir,
                        help='Path to the tensorboard logs directory.')
    # optional config file
    parser.add_argument('-c', '--config-path', type=Path, action='store',
                        metavar='configuration-file',
                        help='Path to the configuration file that will be '
                             'used to set the hyperparameters of the model.')
    # path to the pretrained MLM model that will be used for fine-tuning
    parser.add_argument('-t', '--pre-trained-model', type=Path, action='store',
                        metavar='pre-trained-mlm', default=default_loaddir,
                        help='The path to the pre-trained MLM Ancient Greek '
                             'RoBERTa model.')
    # path to save the directory where the model will be saved
    parser.add_argument('-s', '--savedir', type=Path, action='store',
                        default=default_savedir, metavar='model-save-directory',
                        help='Path to the directory where the PoS tagging '
                             'fine-tuned model will be saved.')
    # optional path to save plot with learning curves
    parser.add_argument('-p', '--plot-savepath', type=Path, action='store',
                        metavar='path-to-save-learning-curves-plot',
                        help='Path to the a .png filename where the learning '
                             'curves for the current experiment will be saved.')
    # optional path to save the confusion matrix of the test set
    parser.add_argument('-m', '--confusion-matrix', type=Path, action='store',
                        metavar='path-to-confusion-matrix-heatmap',
                        help='Path to the .png filename where the confusion '
                             'matrix of the test set will be saved.')
    # whether to omit cuda
    parser.add_argument('-n', '--no-cuda', action='store_true',
                        help='If toggled, then no GPUs will be used for '
                             'training. Else, all available GPUs will be used.')
    # random seed
    parser.add_argument('--seed', type=str, action='store',
                        default=default_seed, metavar='seed',
                        help='Random seed to be set. Used for reproducibility '
                             'purposes. If not specified, then a random seed '
                             'will be set.')

    return parser.parse_args(arg)


def parse_tune_mlm_input(arg: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses the input for the MLM Hyperparameter Tuning scripts (same for
        both frameworks)."""
    default_max_evals = 100

    desc = 'Hyperparameter Tuning script.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-e', '--max-evals', type=int, action='store',
                        metavar='number-of-maximum-objective-'
                                'function-evaluations',
                        default=default_max_evals,
                        help='The number of maximum objective function '
                             'evaluations to perform during the bayesian '
                             'search.')

    return parser.parse_args(arg)
