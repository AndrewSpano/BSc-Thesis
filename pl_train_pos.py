import torch
import argparse
import pytorch_lightning as pl

from transformers import RobertaTokenizerFast
from pytorch_lightning.loggers import TensorBoardLogger

from pl_models.pos_model import PoSRoBERTa
from utils.cmd_args import parse_pl_pos_input
from utils.plot_utils import plot_pos_metrics
from ag_datasets.pos_dataset import PoSDataset
from utils.fs_utils import force_empty_directory
from utils.run_utils import device_from_str, get_seed, hyperparams_from_config
from data_preparation.processing import (
    TOKENIZER_PATH,
    LABEL_ENCODER_PATH,
    PROCESSED_DATA_PATH
)


def main(args: argparse.Namespace):
    """main() driver function."""

    # fix some args
    device_str = device_from_str(args.device)
    if args.distributed is True and device_str == 'cpu':
        raise RuntimeError("Distributed training can needs CUDA.")
    gpus = torch.cuda.device_count() if args.distributed is True else \
        1 if device_str == 'cuda' else None
    distributed_strategy = 'ddp' if args.distributed is True else None
    seed = get_seed(args.seed)

    # empty the tensorboard and model directories
    force_empty_directory(args.logdir)
    force_empty_directory(args.savedir)

    # load the tokenizer and fix the random seed
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    pl.seed_everything(seed)

    # define the default hyperparameters for the model here
    custom_hyperparameters = {
        'max-length': 512,
        'batch-size': 4,
        'learning-rate': 1e-4,
        'weight-decay': 1e-2,
        'use-lr-scheduler': True,
        'scheduler-factor': 0.1,
        'scheduler-patience': 10,
        'scheduler-step-update': 10,
        'train-epochs': 4
    }

    # either use those or load ones from a configuration file
    hyperparams = custom_hyperparameters \
        if args.config_path is None \
        else hyperparams_from_config(args.config_path)

    # create PL model
    data_dir = PROCESSED_DATA_PATH/'PoS'
    data_paths = (
        (data_dir/'pos-train-input-ids.pkl', data_dir/'pos-train-labels.pkl'),
        (data_dir/'pos-val-input-ids.pkl', data_dir/'pos-val-labels.pkl'),
        (data_dir/'pos-test-input-ids.pkl', data_dir/'pos-test-labels.pkl')
    )
    model = PoSRoBERTa(
        mlm_model_path=args.pre_trained_model,
        tokenizer=tokenizer,
        paths=data_paths,
        le_path=LABEL_ENCODER_PATH,
        hyperparams=hyperparams,
        num_classes=PoSDataset.num_classes(LABEL_ENCODER_PATH),
        test_cm_path=args.confusion_matrix
    )

    # train the model
    logger = TensorBoardLogger(str(args.logdir), name='PoS-RoBERTa', version=0)
    trainer = pl.Trainer(
        default_root_dir=str(args.logdir),
        gpus=gpus,
        strategy=distributed_strategy,
        max_epochs=hyperparams['train-epochs'],
        fast_dev_run=False,
        logger=logger,
        log_every_n_steps=1
    )
    trainer.fit(model)
    trainer.test(ckpt_path='best')

    # save the model and (optionally) the learning curves plot
    model.model.save_pretrained(args.savedir)
    if args.plot_savepath is not None:
        plot_pos_metrics(args.logdir, args.plot_savepath, framework='pl')


if __name__ == "__main__":
    print()
    arg = parse_pl_pos_input()
    main(arg)
