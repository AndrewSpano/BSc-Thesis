import torch
import torch.utils.data
import pytorch_lightning as pl

from pathlib import Path
from torch.optim import AdamW
from typing import Tuple, List, Dict, Union, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.modeling_outputs import MaskedLMOutput
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from transformers import RobertaTokenizerFast, RobertaForTokenClassification

from ag_datasets.pos_dataset import PoSDataset
from utils.plot_utils import plot_confusion_matrix


class PoSRoBERTa(pl.LightningModule):
    """Wrapper class for a Lightning Module model."""

    def __init__(
            self,
            mlm_model_path: Path,
            tokenizer: RobertaTokenizerFast,
            paths: Tuple[Tuple[Path, Path], Tuple[Path, Path],
                         Tuple[Path, Path]],
            le_path: Path,
            hyperparams: Dict[str, Union[int, float]],
            num_classes: int,
            test_cm_path: Optional[Path]
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.train_data_path, self.val_data_path, self.test_data_path = paths
        self.model = RobertaForTokenClassification.from_pretrained(
            mlm_model_path, num_labels=num_classes)
        self.freeze_base()
        self.le_path = le_path
        self.hyperparams = hyperparams
        self.num_classes = num_classes
        self.test_cm_path = test_cm_path
        self.val_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.acc = Accuracy(num_classes=num_classes)
        self.f1 = F1Score(num_classes=num_classes, average='weighted')
        self.cm = ConfusionMatrix(num_classes=num_classes)

    def freeze_base(self) -> None:
        for param in self.model.roberta.parameters():
            param.requires_grad = False
        self.model.roberta.eval()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = PoSDataset(
            tokenizer=self.tokenizer,
            input_ids_path=self.train_data_path[0],
            labels_path=self.train_data_path[1],
            le_path=self.le_path,
            maxlen=self.hyperparams['max-length']
        )
        self.val_ds = PoSDataset(
            tokenizer=self.tokenizer,
            input_ids_path=self.val_data_path[0],
            labels_path=self.val_data_path[1],
            le_path=self.le_path,
            maxlen=self.hyperparams['max-length']
        )
        self.test_ds = PoSDataset(
            tokenizer=self.tokenizer,
            input_ids_path=self.test_data_path[0],
            labels_path=self.test_data_path[1],
            le_path=self.le_path,
            maxlen=self.hyperparams['max-length']
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor) -> MaskedLMOutput:
        return self.model(input_ids, attention_mask=attention_mask,
                          labels=labels)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) \
            -> Dict[str, torch.Tensor]:
        outputs = self.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits.view(-1, self.num_classes)

        labels = batch['labels'].view(-1)
        pred_labels = torch.argmax(logits, dim=1)

        valid_indices = labels != -100
        labels = labels[valid_indices]
        pred_labels = pred_labels[valid_indices]

        acc = self.acc(pred_labels, labels)
        f1 = self.f1(pred_labels, labels)

        self.log('train/batch_loss', loss.item())
        self.log('train/batch_acc', acc)
        self.log('train/batch_f1', f1)

        return {'loss': loss, 'acc': acc, 'f1': f1}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_id: int) \
            -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.forward(**batch)
        logits = outputs.logits.view(-1, self.num_classes)
        loss = self.val_criterion(logits, batch['labels'].view(-1))

        labels = batch['labels'].view(-1)
        pred_labels = torch.argmax(logits, dim=1)

        valid_indices = labels != -100
        labels = labels[valid_indices]
        pred_labels = pred_labels[valid_indices]

        return {'loss': loss, 'labels': labels, 'pred_labels': pred_labels}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) \
            -> Dict[str, float]:
        """Computes the average batch loss and returns it along with logging
            information."""
        loss = torch.cat([o['loss'] for o in outputs], dim=0).mean().item()
        all_labels = torch.cat([o['labels'] for o in outputs], dim=0)
        all_preds = torch.cat([o['pred_labels'] for o in outputs], dim=0)
        acc = self.acc(all_preds, all_labels)
        f1 = self.f1(all_preds, all_labels)

        self.log('val/val_loss', loss)
        self.log('val/val_acc', acc)
        self.log('val/val_f1', f1)

        return {'loss': loss, 'acc': acc, 'f1': f1}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) \
            -> Dict[str, torch.Tensor]:
        """Computes the test loss for the current batch and returns it."""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) \
            -> Dict[str, float]:
        """Computes the average test metrics and logs them."""
        test_loss = torch.cat([o['loss'] for o in outputs], 0).mean().item()
        all_labels = torch.cat([o['labels'] for o in outputs], dim=0)
        all_preds = torch.cat([o['pred_labels'] for o in outputs], dim=0)
        acc = self.acc(all_preds, all_labels)
        f1 = self.f1(all_preds, all_labels)

        self.log('test/test_loss', test_loss)
        self.log('test/test_acc', acc)
        self.log('test/test_f1', f1)

        cm = self.cm(all_preds, all_labels)
        classes = self.test_ds.classnames
        plot_confusion_matrix(cm, classes, self.test_cm_path)

        return {'loss': test_loss, 'acc': acc, 'f1': f1}

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
        """Return the optimizer and the learning rate scheduler (if used)."""
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
