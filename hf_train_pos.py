import glob
import torch
import argparse

from transformers import (
    RobertaTokenizerFast,
    TrainingArguments,
    IntervalStrategy,
    SchedulerType,
    RobertaForTokenClassification,
    Trainer,
    set_seed
)
from typing import Dict, Tuple
from torchmetrics import ConfusionMatrix
from sklearn.metrics import accuracy_score, f1_score
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import EvalPrediction

from utils.cmd_args import parse_hf_pos_input
from ag_datasets.pos_dataset import PoSDataset
from utils.fs_utils import force_empty_directory
from utils.run_utils import hyperparams_from_config, get_seed
from utils.plot_utils import plot_pos_metrics, plot_confusion_matrix
from data_preparation.processing import (
    TOKENIZER_PATH,
    PROCESSED_DATA_PATH,
    LABEL_ENCODER_PATH
)


class CustomMetricsTrainer(Trainer):
    """Overriding the Trainer Class so that custom metrics such as Accuracy
        and F1 score can be logged during training."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override the compute_loss() function such that it logs the
            accuracy and the f1 score."""
        if self.label_smoother is not None and 'labels' in inputs:
            labels = inputs.pop('labels')
        else:
            labels = None
        outputs = model(**inputs)

        # compute batch accuracy and f1 score for training batches
        #  Small hack: If the logits do not require a gradient, then this
        #    function has been called with torch.no_grad(), which means that
        #    this is an evaluation call, so don't compute the metrics as this
        #    block is meant only for training.
        if 'labels' in inputs and outputs.logits.requires_grad:
            preds = outputs.logits.detach().cpu().argmax(-1).reshape(-1).numpy()
            labels_ = inputs['labels'].detach().cpu().reshape(-1).numpy()

            valid_indices = labels_ != -100
            preds = preds[valid_indices]
            labels_ = labels_[valid_indices]

            acc = accuracy_score(labels_, preds)
            f1 = f1_score(labels_, preds, average='weighted')

            self.log({'accuracy': acc, 'f1': f1})

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples
            #  instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def main(args: argparse.Namespace):
    """main() driver function."""

    # args
    seed = get_seed(args.seed)
    set_seed(seed)

    # empty the tensorboard and model directories
    force_empty_directory(args.logdir)
    force_empty_directory(args.savedir)

    # create the model
    model_dir = glob.glob(f'{args.pre_trained_model}/checkpoint-*')[0]
    model = RobertaForTokenClassification.from_pretrained(
        model_dir,
        num_labels=PoSDataset.num_classes(LABEL_ENCODER_PATH)
    )

    # define the custom hyperparameters for the model here
    custom_hyperparameters = {
        'max-length': 512,
        'batch-size': 4,
        'learning-rate': 1e-4,
        'weight-decay': 1e-2,
        'decay-lr-at-percentage-of-steps': 0.1,
        'train-epochs': 5
    }

    # either use those or load ones from a configuration file
    hyperparams = custom_hyperparameters \
        if args.config_path is None \
        else hyperparams_from_config(args.config_path)

    # load the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)

    # create datasets
    data_dir = PROCESSED_DATA_PATH/'PoS'
    train_dataset = PoSDataset(
        tokenizer=tokenizer,
        input_ids_path=data_dir/'pos-train-input-ids.pkl',
        labels_path=data_dir/'pos-train-labels.pkl',
        le_path=LABEL_ENCODER_PATH,
        maxlen=hyperparams['max-length']
    )
    val_dataset = PoSDataset(
        tokenizer=tokenizer,
        input_ids_path=data_dir/'pos-val-input-ids.pkl',
        labels_path=data_dir/'pos-val-labels.pkl',
        le_path=LABEL_ENCODER_PATH,
        maxlen=hyperparams['max-length']
    )
    test_dataset = PoSDataset(
        tokenizer=tokenizer,
        input_ids_path=data_dir/'pos-test-input-ids.pkl',
        labels_path=data_dir/'pos-test-labels.pkl',
        le_path=LABEL_ENCODER_PATH,
        maxlen=hyperparams['max-length']
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

    # define a function that return the logits/labels without padding entries
    def unpad(labels_: torch.Tensor, preds_: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """Removes values where the label is -100 and returns both Tensors."""
        valid_indices = labels_ != -100
        return labels_[valid_indices], preds_[valid_indices]

    # define the metrics used (accuracy and F1)
    def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
        """Computes some metrics given the predictions and labels, and returns
            them in the dictionary so that they can be digested by the HF
            Trainer API."""
        labels_ = pred.label_ids.reshape(-1)
        preds_ = pred.predictions.reshape(-1)
        labels_, preds_ = unpad(labels_, preds_)

        acc_ = accuracy_score(labels_, preds_)
        f1_ = f1_score(labels_, preds_, average='weighted')
        return {'accuracy': acc_, 'f1': f1_}

    # train
    trainer = CustomMetricsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, _: logits.argmax(-1)
    )
    trainer.train()

    # get the test metrics
    test_out = trainer.predict(test_dataset=test_dataset)

    if args.confusion_matrix is not None:
        labels = torch.from_numpy(test_out.label_ids)
        preds = torch.from_numpy(test_out.predictions)
        labels, preds = unpad(labels, preds)
        classes = test_dataset.classnames
        cm = ConfusionMatrix(num_classes=len(classes))(preds, labels)
        plot_confusion_matrix(cm, classes, args.confusion_matrix)

    test_loss, acc, f1 = (test_out.metrics['test_loss'],
                          test_out.metrics['test_accuracy'],
                          test_out.metrics['test_f1'])
    print(f'Test Loss: {test_loss:.6f}\n'
          f'Test Accuracy: {acc:.2f}\n'
          f'Test weighted F1 score: {f1:.2f}')
    test_metrics = (test_loss, acc, f1)

    # save plots with losses if specified
    if args.plot_savepath is not None:
        plot_pos_metrics(args.logdir, args.plot_savepath,
                         framework='hf', test_metrics=test_metrics)


if __name__ == "__main__":
    print()
    arg = parse_hf_pos_input()
    main(arg)
