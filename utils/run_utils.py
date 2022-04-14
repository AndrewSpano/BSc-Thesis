import ast
import torch
import random
import configparser

from pathlib import Path
from typing import Dict, Union


def device_from_str(device_str: str) -> str:
    """Fixes the torch device string if needed."""
    if device_str == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def get_seed(seed_str: str) -> int:
    if seed_str == 'random':
        return random.randrange(int(1e10))
    else:
        return int(seed_str)


def hyperparams_from_config(config_path: Path) \
        -> Dict[str, Union[int, float, bool]]:
    """Given the path of a config file, it reads it and returns the
        hyperparameters it contains in a dictionary-like object."""
    def values_to_numeric(inp: Dict[str, str]) -> Dict[str, Union[int, float]]:
        """Converts the values of a dictionary from string to numerical."""
        return {key: ast.literal_eval(value) for key, value in inp.items()}

    cf = configparser.ConfigParser()
    cf.read(config_path)
    defaults = values_to_numeric(dict(cf['defaults']))
    hyperparameters = values_to_numeric(dict(cf['hyperparameters']))
    return {**defaults, **hyperparameters}
