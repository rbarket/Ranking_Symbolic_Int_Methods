import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from src.data.dataset import PrefixExpressionDataset, collate_fn

def get_dataloader(
    cfg,
    split: str | list[str] = "train",
    sample_n: int = None
):
    if isinstance(split, str):
        ds = PrefixExpressionDataset(cfg, split=split, sample_n=sample_n)
    else:
        ds = ConcatDataset([PrefixExpressionDataset(cfg, split=s, sample_n=sample_n) for s in split])
        
    return DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
)