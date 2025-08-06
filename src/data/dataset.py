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
    file_path = input_dir / f"{split}_data_reg.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find split file: {file_path}")
    return pd.read_parquet(file_path)

class PrefixExpressionDataset(Dataset):
    def __init__(self, cfg, split="train", sample_n: int = None):
        """
        Args:
          cfg: DotDict config
          split: "train"/"test"/etc.
          sample_n: if not None and split=="train", randomly sample this many rows
        """
        # 1. Load full DataFrame
        df = load_split(cfg, split)

        # 2. Optional subsampling BEFORE any token/positional work
        if split == "train" and sample_n is not None:
            df = df.sample(n=min(sample_n, len(df)), random_state=1998)

        # 3. Load vocab & positional encodings
        self.vocab = load_vocab(cfg)
        self.positions = load_precomputed_positions(cfg)

        # 4. Params
        self.n = cfg.tree.branching_factor
        self.k = cfg.tree.depth

        # 5. Extract expressions, labels, masks
        expressions = df["prefix"].tolist()
        labels_raw   = df["label"].tolist()
        masks_raw    = [[l != -1 for l in lbls] for lbls in labels_raw]

        # 6. Build the data tuples once, now on the (possibly smaller) DataFrame
        self.data = []
        for expr, label_list, mask_list in zip(expressions, labels_raw, masks_raw):
            # Token IDs + tree paths
            tokens, path_list = get_prefix_data_with_paths(expr)
            token_ids = torch.tensor(
                [ self.vocab.get(tok, 1) for tok in tokens ],
                dtype=torch.long
            )

            # Positional encodings per token
            pos_encs = []
            for path in path_list:
                level = min(len(path), self.k)
                idx   = path_to_index(path[:level])
                pos_encs.append(self.positions[level][idx])
            pos_tensor = torch.stack(pos_encs)  # [T, d_model]

            # Labels & mask tensors
            label_tensor = torch.tensor(label_list, dtype=torch.float32)
            mask_tensor  = torch.tensor(mask_list,  dtype=torch.bool)

            self.data.append((token_ids, pos_tensor, label_tensor, mask_tensor))

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