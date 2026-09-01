import torch
import yaml
from copy import deepcopy

class DotDict:
    """
    Dictionary wrapper that supports attribute-style access,
    recursively converting nested dicts to DotDict.
    """
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                v = DotDict(v)
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"DotDict({self.__dict__})"

    def to_dict(self) -> dict:
        return config_to_dict(self)


def config_to_dict(value):
    """Recursively convert configuration objects into checkpoint-safe values."""
    if isinstance(value, DotDict):
        value = value.__dict__
    elif hasattr(value, "__dict__"):
        value = vars(value)
    if isinstance(value, dict):
        return {key: config_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_to_dict(item) for item in value]
    return deepcopy(value)


def config_from_dict(values: dict) -> DotDict:
    if not isinstance(values, dict):
        raise TypeError("A checkpoint configuration snapshot must be a dictionary.")
    return DotDict(deepcopy(values))


def load_config(path: str) -> DotDict:
    """
    Load a YAML configuration file and return it as a DotDict,
    allowing nested attribute access (e.g., cfg.data.batch_size).

    Args:
        path (str): Path to the YAML config file.

    Returns:
        DotDict: Configuration object with attribute access.
    """
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return DotDict(cfg_dict)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve ``auto``/CPU/CUDA CLI values with a safe CPU fallback."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        return torch.device(device_arg if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
