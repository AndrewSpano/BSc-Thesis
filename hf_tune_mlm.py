import argparse

from transformers import (
    RobertaTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    IntervalStrategy,
    SchedulerType,
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer
)
from ray import tune
from pathlib import Path
from ray.tune.trial import Trial
from typing import Dict, Optional, Any
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import HPSearchBackend

from utils.cmd_args import parse_tune_mlm_input
from ag_datasets.hf_mlm_dataset import AGHFMLMDataset
from utils.fs_utils import force_empty_directory, delete_file_if_exists
from data_preparation.processing import TOKENIZER_PATH, PROCESSED_DATA_PATH


def main(args: argparse.Namespace):
    """main() driver function."""

    # define the constant values of the model
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
    local_dir = (Path('logs')/'hf-mlm-ray-tune-results').absolute()
    force_empty_directory(local_dir)
    output_dir = Path('objects')/'HF-Tuned-AG-RoBERTa'
    force_empty_directory(output_dir)
    tune_logfile = (Path('logs')/'hf-mlm-hp-tuning-results.txt').absolute()
    delete_file_if_exists(tune_logfile)
    resources_per_trial = {'cpu': 1, 'gpu': 1}
    constants = {
        'max-length': 512,
        'mask-probability': 0.15,
        'type-vocab-size': 1,
        'decay-lr-at-percentage-of-steps': 0.1,
        'train-epochs': 2,
        'tokenizer': tokenizer,
        'local-dir': local_dir,
        'output-dir': output_dir,
        'tune-logfile': tune_logfile,
        'resources-per-trial': resources_per_trial
    }

    def model_init(trial: Trial) -> RobertaForMaskedLM:
        """Initializes and returns a model given a Ray Tune trial object."""
        # trial will be `None` during the creation of Trainer()
        if trial is None:
            trial = {
                'hidden-size': 128,
                'num-attention-heads': 2,
                'num-hidden-layers': 2
            }

        # the hidden size must be a multiple of the number of attention heads
        hidden_size = trial['hidden-size']
        num_attention_heads = trial['num-attention-heads']
        hidden_size = (hidden_size // num_attention_heads) * num_attention_heads

        # create and return the model
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=constants['max-length'] + 2,
            hidden_size=int(hidden_size),
            num_attention_heads=int(trial['num-attention-heads']),
            num_hidden_layers=int(trial['num-hidden-layers']),
            type_vocab_size=constants['type-vocab-size'],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        model = RobertaForMaskedLM(config).train()
        return model

    # define the hyperparameter search space
    def search_space(_: Optional[Any] = None) -> \
            Dict[str, float]:
        """Returns the ray tune search space used for hyperparameter search."""
        return {
            'hidden-size': tune.choice([256, 512, 768, 1024]),
            'num-attention-heads': tune.quniform(2, 16, 1),
            'num-hidden-layers': tune.quniform(2, 12, 1),
            'per_device_train_batch_size': tune.choice([4, 8, 16, 32]),
            'learning_rate': tune.loguniform(1e-6, 3e-4),
            'weight_decay': tune.loguniform(1e-2, 1),
            'seed': tune.choice([3, 13, 420, 3407, 80085])
        }

    # create datasets
    data_dir = PROCESSED_DATA_PATH/'MLM'
    train_dataset = AGHFMLMDataset(data_dir/'train-data.pkl')
    val_dataset = AGHFMLMDataset(data_dir/'val-data.pkl')

    # load the tokenizer and create the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=constants['mask-probability']
    )

    # train args
    training_args = TrainingArguments(
        output_dir=str(constants['output-dir']),
        overwrite_output_dir=True,
        evaluation_strategy=IntervalStrategy.EPOCH,
        prediction_loss_only=False,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=1,
        num_train_epochs=constants['train-epochs'],
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_ratio=constants['decay-lr-at-percentage-of-steps'],
        log_level='passive',
        logging_strategy=IntervalStrategy.STEPS,
        logging_first_step=True,
        logging_steps=1,
        save_strategy=IntervalStrategy.EPOCH,
        save_total_limit=1,
        no_cuda=False,
        local_rank=-1,
        dataloader_drop_last=False,
        dataloader_num_workers=1,
        optim=OptimizerNames.ADAMW_TORCH,
        group_by_length=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        skip_memory_metrics=True,
        disable_tqdm=True
    )

    # create a Trainer object and perform hyperparameter search
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    # if no `compute_objective` function is provided, then the validation loss
    #  is chosen by default as the objective
    best_run = trainer.hyperparameter_search(
        hp_space=search_space,
        n_trials=args.max_evals,
        direction='minimize',
        backend=HPSearchBackend.RAY,
        search_alg=HyperOptSearch(metric='objective', mode='min'),
        scheduler=ASHAScheduler(metric='objective', mode='min'),
        resources_per_trial=constants['resources-per-trial'],
        local_dir=str(constants['local-dir']),
        log_to_file=str(constants['tune-logfile'])
    )
    print(f'Best hyperparameter combination found: {best_run.hyperparameters}.'
          f'\nValidation loss achieved: {best_run.objective}')


if __name__ == "__main__":
    print()
    arg = parse_tune_mlm_input()
    main(arg)
