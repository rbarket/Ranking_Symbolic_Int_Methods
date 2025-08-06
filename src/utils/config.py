import yaml

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
