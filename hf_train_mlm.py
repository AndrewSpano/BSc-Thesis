import torch
import argparse

from transformers import (
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    IntervalStrategy,
    SchedulerType,
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer,
    set_seed
)
from transformers.training_args import OptimizerNames

from utils.plot_utils import plot_mlm_losses
from utils.cmd_args import parse_hf_mlm_input
from utils.fs_utils import force_empty_directory
from ag_datasets.hf_mlm_dataset import AGHFMLMDataset
from utils.run_utils import hyperparams_from_config, get_seed
from data_preparation.processing import TOKENIZER_PATH, PROCESSED_DATA_PATH


def main(args: argparse.Namespace):
    """main() driver function."""

    # args
    seed = get_seed(args.seed)
    set_seed(seed)

    # empty the tensorboard and model directories
    force_empty_directory(args.logdir)
    force_empty_directory(args.savedir)

    # define the custom hyperparameters for the model here
    custom_hyperparameters = {
        'max-length': 512,
        'batch-size': 4,
        'mask-probability': 0.15,
        'hidden-size': 32,
        'num-attention-heads': 2,
        'num-hidden-layers': 2,
        'type-vocab-size': 1,
        'learning-rate': 1e-4,
        'weight-decay': 1e-2,
        'decay-lr-at-percentage-of-steps': 0.1,
        'train-epochs': 10
    }

    # either use those or load ones from a configuration file
    hyperparams = custom_hyperparameters \
        if args.config_path is None \
        else hyperparams_from_config(args.config_path)

    # create datasets
    data_dir = PROCESSED_DATA_PATH/'MLM'
    train_dataset = AGHFMLMDataset(data_dir/'train-data.pkl')
    val_dataset = AGHFMLMDataset(data_dir/'val-data.pkl')
    test_dataset = AGHFMLMDataset(data_dir/'test-data.pkl')

    # load the tokenizer and create the data collator
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=hyperparams['mask-probability']
    )

    # train args
    training_args = TrainingArguments(
        output_dir=args.savedir,
        overwrite_output_dir=True,
        evaluation_strategy=IntervalStrategy.EPOCH,
        prediction_loss_only=False,
        per_device_train_batch_size=hyperparams['batch-size'],
        per_device_eval_batch_size=hyperparams['batch-size'],
        learning_rate=hyperparams['learning-rate'],
        weight_decay=hyperparams['weight-decay'],
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=1,
        num_train_epochs=hyperparams['train-epochs'],
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_ratio=hyperparams['decay-lr-at-percentage-of-steps'],
        log_level='passive',
        logging_dir=args.logdir,
        logging_strategy=IntervalStrategy.STEPS,
        logging_first_step=True,
        logging_steps=1,
        save_strategy=IntervalStrategy.EPOCH,
        save_total_limit=1,
        no_cuda=args.no_cuda,
        seed=seed,
        local_rank=-1,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        optim=OptimizerNames.ADAMW_TORCH,
        group_by_length=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        skip_memory_metrics=True
    )

    # create the model
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=hyperparams['max-length'] + 2,
        hidden_size=hyperparams['hidden-size'],
        num_attention_heads=hyperparams['num-attention-heads'],
        num_hidden_layers=hyperparams['num-hidden-layers'],
        type_vocab_size=hyperparams['type-vocab-size'],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    model = RobertaForMaskedLM(config).train()

    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=lambda _, __: torch.Tensor([])
    )
    trainer.train()

    # get the test loss
    test_out = trainer.predict(test_dataset=test_dataset)
    test_loss = test_out.metrics['test_loss']
    print(f'Test Loss: {test_loss:.6f}')

    # save plots with losses if specified
    if args.plot_savepath is not None:
        plot_mlm_losses(args.logdir, args.plot_savepath, framework='hf',
                        test_loss=test_loss)


if __name__ == "__main__":
    print()
    arg = parse_hf_mlm_input()
    main(arg)
