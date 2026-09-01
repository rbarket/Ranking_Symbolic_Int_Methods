import json
from torch import load
from pathlib import Path


def load_vocab_path(path) -> dict:
    """Load a token-to-index vocabulary directly from a JSON path."""
    vocab_file = Path(path)
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
    with vocab_file.open() as file:
        return json.load(file)

def load_vocab(cfg):
    """
    Load the token‐to‐index mapping from disk.
    """
    embedded_vocab = getattr(cfg.data, "embedded_vocab", None)
    if embedded_vocab is not None:
        return dict(embedded_vocab)
    return load_vocab_path(cfg.data.vocab_path)

def load_precomputed_positions(cfg):
    """
    Load the precomputed positional-encoding tensors from disk.
    Assumes they were saved next to vocab.json as 'precomputed_positions.pt'.
    """
    configured_path = getattr(cfg.data, "positions_path", None)
    if configured_path is not None:
        positions_file = Path(configured_path)
    else:
        vocab_file = Path(cfg.data.vocab_path)
        positions_file = vocab_file.parent / "precomputed_positions.pt"
    if not positions_file.exists():
        raise FileNotFoundError(f"Positional encodings not found: {positions_file}")
    return load(positions_file)
