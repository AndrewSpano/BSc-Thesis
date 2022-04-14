import torch
import pickle
import torch.utils.data

from typing import Dict
from pathlib import Path
from transformers import RobertaTokenizerFast

from ag_datasets.dataset_utils import mlm


class AGMLMDataset(torch.utils.data.Dataset):
    """Implements a torch Dataset class for Ancient Greek input data that
        will be fed to a RoBERTa model. Uses dynamic masking."""

    def __init__(
            self,
            input_ids_path: Path,
            tokenizer: RobertaTokenizerFast,
            mask_probability: float,
            max_length: int
    ) -> None:
        """
        :param input_ids_path:
            Path to the pickle file containing the input IDs for MLM.

        :param tokenizer:
            A RobertaTokenizerFast object.

        :param mask_probability:
            The masking probability used for dynamic masking.

        :param max_length:
            Maximum length of sequence.
        """
        with open(input_ids_path, 'rb') as fp:
            self.input_ids = pickle.load(fp)
        self.tokenizer = tokenizer
        self.p = mask_probability
        self.maxlen = max_length

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        pads_needed = self.maxlen - len(self.input_ids[item])
        input_ids = torch.tensor(self.input_ids[item] +
                                 [self.tokenizer.pad_token_id] * pads_needed)
        return {
            # implements dynamic masking as in the original RoBERTa paper
            'input_ids': mlm(input_ids, self.tokenizer, self.p).squeeze(),
            'attention_mask': input_ids != self.tokenizer.pad_token_id,
            'labels': input_ids
        }
