import torch
import pickle
import torch.utils.data

from typing import Dict
from pathlib import Path


class AGHFMLMDataset(torch.utils.data.Dataset):
    """Implements a torch Dataset class for Ancient Greek input data that
        will be used by the HF Trainer API during training."""

    def __init__(
            self,
            input_ids_path: Path,
    ) -> None:
        """
        :param input_ids_path:
            Path to the .pkl file containing encoded (by the tokenizer)
            input sentences for the model.
        """
        with open(input_ids_path, 'rb') as fp:
            input_ids = pickle.load(fp)
        self.input_ids = [{'input_ids': torch.tensor(e, dtype=torch.long)}
                          for e in input_ids]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        return self.input_ids[item]
