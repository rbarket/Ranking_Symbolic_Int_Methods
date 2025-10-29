import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import pandas as pd

from src.utils.tree_utils import get_prefix_data_with_paths, path_to_index
from src.utils.io import load_precomputed_positions, load_vocab

def load_split(cfg, split: str = "train") -> pd.DataFrame:
    """
    Load an entire dataset split (train/test) as a pandas DataFrame.
    """
    input_dir = Path(cfg.data.input_dir)
    file_path = input_dir / f"{split}_data.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find split file: {file_path}")
    return pd.read_parquet(file_path)


class PrefixExpressionDataset(Dataset):
    def __init__(self, cfg, split="train", sample_n=None):
        self.df_all = load_split(cfg, split)  # load everything once
        self.cfg = cfg
        self.split = split
        self.sample_n = sample_n
        self.set_data_type(None)  # default: no filter

    def set_data_type(self, data_type):
        """Change data type filter and rebuild `self.data` without reloading file"""
        if data_type is None:
            df = self.df_all
        else:
            df = self.df_all[self.df_all["source"] == data_type]

        if self.split == "train" and self.sample_n is not None:
            df = df.sample(n=min(self.sample_n, len(df)), random_state=1998)

        self._build_data(df)

    def _build_data(self, df):
        self.vocab = load_vocab(self.cfg)
        self.positions = load_precomputed_positions(self.cfg)
        self.n = self.cfg.tree.branching_factor
        self.k = self.cfg.tree.depth

        expressions = df["prefix"].tolist()
        labels_raw  = df["label"].tolist()
        masks_raw   = [[l != -1 for l in lbls] for lbls in labels_raw]

        self.data = []
        vocab_get = self.vocab.get
        positions = self.positions
        k = self.k
        append_data = self.data.append

        for expr, label_list, mask_list in zip(expressions, labels_raw, masks_raw):
            tokens, path_list = get_prefix_data_with_paths(expr)
            token_ids = torch.tensor([vocab_get(tok, 1) for tok in tokens], dtype=torch.long)
            # for each token, get the precomputed position encoding based on its path
            pos_tensor = torch.stack([
                positions[min(len(path), k)][path_to_index(path[:min(len(path), k)])]
                for path in path_list
            ])
            label_tensor = torch.tensor(label_list, dtype=torch.float32)
            mask_tensor  = torch.tensor(mask_list, dtype=torch.bool)
            append_data((token_ids, pos_tensor, label_tensor, mask_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    token_ids, pos_encs, labels, masks = zip(*batch)
    padded_tokens = pad_sequence(token_ids, batch_first=True, padding_value=0)
    padded_pos    = pad_sequence(pos_encs,   batch_first=True, padding_value=0.0)
    token_mask    = padded_tokens.eq(0)
    label_tensor  = torch.stack(labels)
    mask_tensor   = torch.stack(masks)
    return padded_tokens, padded_pos, token_mask, label_tensor, mask_tensor

# testing