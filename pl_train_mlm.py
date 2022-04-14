import torch
import argparse
import pytorch_lightning as pl

from transformers import RobertaTokenizerFast
from pytorch_lightning.loggers import TensorBoardLogger

from utils.plot_utils import plot_mlm_losses
from utils.cmd_args import parse_pl_mlm_input
from pl_models.mlm_model import LitRoBERTaMLM
from utils.fs_utils import force_empty_directory
from data_preparation.processing import TOKENIZER_PATH, PROCESSED_DATA_PATH
from utils.run_utils import device_from_str, get_seed, hyperparams_from_config


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
        'mask-probability': 0.15,
        'hidden-size': 768,
        'num-attention-heads': 12,
        'num-hidden-layers': 6,
        'type-vocab-size': 1,
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
    data_dir = PROCESSED_DATA_PATH/'MLM'
    data_paths = (data_dir/'train-data.pkl',
                  data_dir/'val-data.pkl',
                  data_dir/'test-data.pkl')
    model = LitRoBERTaMLM(
        tokenizer=tokenizer,
        paths=data_paths,
        hyperparams=hyperparams
    )

    # train the model
    logger = TensorBoardLogger(str(args.logdir), name='AG-RoBERTa', version=0)
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
        plot_mlm_losses(args.logdir, args.plot_savepath, framework='pl')


if __name__ == "__main__":
    print()
    arg = parse_pl_mlm_input()
    main(arg)
