import torch
import torch.utils.data
import pytorch_lightning as pl

from pathlib import Path
from torch.optim import AdamW
from typing import List, Tuple, Dict, Union, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM
)
from transformers.modeling_outputs import MaskedLMOutput

from ag_datasets.mlm_dataset import AGMLMDataset


def create_roberta_mlm_model(
        roberta_tokenizer: RobertaTokenizerFast,
        hyperparameters: Dict[str, Union[int, float]]
) -> RobertaForMaskedLM:
    """Creates and returns a RoBERTa for MLM model from the given arguments."""
    config = RobertaConfig(
        vocab_size=roberta_tokenizer.vocab_size,
        max_position_embeddings=hyperparameters['max-length'] + 2,
        hidden_size=int(hyperparameters['hidden-size']),
        num_attention_heads=int(hyperparameters['num-attention-heads']),
        num_hidden_layers=int(hyperparameters['num-hidden-layers']),
        type_vocab_size=hyperparameters['type-vocab-size'],
        bos_token_id=roberta_tokenizer.bos_token_id,
        eos_token_id=roberta_tokenizer.eos_token_id,
        pad_token_id=roberta_tokenizer.pad_token_id
    )
    return RobertaForMaskedLM(config).train()


class LitRoBERTaMLM(pl.LightningModule):
    """Wrapper class for a Lightning Module Ancient Greek RoBERTa model."""

    def __init__(
            self,
            tokenizer: RobertaTokenizerFast,
            paths: Tuple[Path, Path, Path],
            hyperparams: Dict[str, Union[int, float]]
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.train_data_path, self.val_data_path, self.test_data_path = paths
        self.model = create_roberta_mlm_model(tokenizer, hyperparams)
        self.hyperparams = hyperparams
        self.val_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the train/val/test datasets."""
        self.train_ds = AGMLMDataset(
            input_ids_path=self.train_data_path,
            tokenizer=self.tokenizer,
            mask_probability=self.hyperparams['mask-probability'],
            max_length=self.hyperparams['max-length']
        )
        self.val_ds = AGMLMDataset(
            input_ids_path=self.val_data_path,
            tokenizer=self.tokenizer,
            mask_probability=self.hyperparams['mask-probability'],
            max_length=self.hyperparams['max-length']
        )
        self.test_ds = AGMLMDataset(
            input_ids_path=self.test_data_path,
            tokenizer=self.tokenizer,
            mask_probability=self.hyperparams['mask-probability'],
            max_length=self.hyperparams['max-length']
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor) -> MaskedLMOutput:
        return self.model(input_ids, attention_mask=attention_mask,
                          labels=labels)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) \
            -> Dict[str, torch.Tensor]:
        """Forward input through the model and returns the loss along with
            logging information."""
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log('train/batch_loss', loss.item())
        return {'loss': loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) \
            -> Dict[str, torch.Tensor]:
        """Computes the validation loss for the current batch and returns it."""
        with torch.no_grad():
            outputs = self.forward(**batch)
        logits = outputs.logits.view(-1, self.model.config.vocab_size)
        loss = self.val_criterion(logits, batch['labels'].view(-1))
        return {'loss': loss}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) \
            -> Dict[str, float]:
        """Computes the average validation loss and returns it along
            with logging information."""
        avg_val_loss = torch.cat([o['loss'] for o in outputs], 0).mean().item()
        self.log('val/val_loss', avg_val_loss)
        return {'loss': avg_val_loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) \
            -> Dict[str, torch.Tensor]:
        """Computes the test loss for the current batch and returns it."""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) \
            -> Dict[str, float]:
        """Computes the average test loss and returns it along
            with logging information."""
        avg_test_loss = torch.cat([o['loss'] for o in outputs], 0).mean().item()
        self.log('test/test_loss', avg_test_loss)
        return {'loss': avg_test_loss}

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.hyperparams['batch-size'],
            shuffle=True,
            num_workers=1
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.hyperparams['batch-size'],
            shuffle=False,
            num_workers=1
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.hyperparams['batch-size'],
            shuffle=False,
            num_workers=1
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()

    def configure_optimizers(self) -> Union[AdamW, Dict[
        str, Union[AdamW, Dict[str, Union[ReduceLROnPlateau, str, int]]]]
    ]:
        """Return the optimizer and the learning rate scheduler
            (if specified)."""
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.hyperparams['learning-rate'],
            weight_decay=self.hyperparams['weight-decay']
        )
        if not self.hyperparams['use-lr-scheduler']:
            return optimizer

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=self.hyperparams['scheduler-factor'],
            patience=self.hyperparams['scheduler-patience'],
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train/batch_loss',
                'interval': 'step',
                'frequency': self.hyperparams['scheduler-step-update']
            }
        }
