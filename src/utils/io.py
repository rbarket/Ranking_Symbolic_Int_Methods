import json
from torch import load
from pathlib import Path

def load_vocab(cfg):
    """
    Load the token‐to‐index mapping from disk.
    """
    vocab_file = Path(cfg.data.vocab_path)
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
    with open(vocab_file, 'r') as f:
        return json.load(f)

def load_precomputed_positions(cfg):
    """
    Load the precomputed positional-encoding tensors from disk.
    Assumes they were saved next to vocab.json as 'precomputed_positions.pt'.
    """
    vocab_file = Path(cfg.data.vocab_path)
    positions_file = vocab_file.parent / "precomputed_positions.pt"
    if not positions_file.exists():
        raise FileNotFoundError(f"Positional encodings not found: {positions_file}")
    return load(positions_file)