from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataset import (
    PrefixExpressionDataset,
    TokenSequenceDataset,
    collate_fn,
    load_split,
    token_sequence_collate_fn,
)
from src.data.tree_lstm_dataset import TreeLSTMGraphDataset, tree_lstm_collate_fn


def split_train_validation(cfg):
    """Return deterministic, disjoint train and validation DataFrames."""
    dataframe = load_split(cfg, split="train")
    validation_fraction = float(
        getattr(cfg.training, "validation_fraction", 0.1)
    )
    split_seed = int(getattr(cfg.training, "split_seed", 1998))

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("training.validation_fraction must be between 0 and 1.")
    if len(dataframe) < 2:
        raise ValueError("At least two training rows are required for a validation split.")

    validation_size = round(len(dataframe) * validation_fraction)
    validation_size = min(max(validation_size, 1), len(dataframe) - 1)
    generator = np.random.default_rng(split_seed)
    permutation = generator.permutation(len(dataframe))

    validation_indices = np.sort(permutation[:validation_size])
    train_indices = np.sort(permutation[validation_size:])
    return dataframe.iloc[train_indices], dataframe.iloc[validation_indices]


def _build_dataloader(
    cfg,
    dataset_class,
    batch_collator,
    split="train",
    sample_n=None,
    dataframe=None,
    pin_memory=None,
    shuffle=None,
):
    if isinstance(split, str):
        dataset = dataset_class(
            cfg,
            split=split,
            sample_n=sample_n,
            dataframe=dataframe,
        )
    else:
        dataset = ConcatDataset(
            [
                dataset_class(cfg, split=item, sample_n=sample_n)
                for item in split
            ]
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=pin_memory,
        collate_fn=batch_collator,
    )

def get_dataloader(
    cfg,
    split: str | list[str] = "train",
    sample_n: int = None,
    dataframe=None,
    shuffle=None,
):
    """Build the legacy tree-position loader used by TreeTransformer."""
    return _build_dataloader(
        cfg,
        PrefixExpressionDataset,
        collate_fn,
        split=split,
        sample_n=sample_n,
        dataframe=dataframe,
        shuffle=shuffle,
    )


def get_token_dataloader(
    cfg,
    split: str | list[str] = "train",
    sample_n: int = None,
    dataframe=None,
    shuffle=None,
):
    return _build_dataloader(
        cfg,
        TokenSequenceDataset,
        token_sequence_collate_fn,
        split=split,
        sample_n=sample_n,
        dataframe=dataframe,
        shuffle=shuffle,
    )


def get_tree_lstm_dataloader(
    cfg,
    split: str | list[str] = "train",
    sample_n: int = None,
    dataframe=None,
    shuffle=None,
):
    return _build_dataloader(
        cfg,
        TreeLSTMGraphDataset,
        tree_lstm_collate_fn,
        split=split,
        sample_n=sample_n,
        dataframe=dataframe,
        pin_memory=False,
        shuffle=shuffle,
    )


def _build_train_validation_dataloaders(
    cfg,
    dataloader_builder,
    train_n=None,
    eval_n=None,
):
    train_frame, validation_frame = split_train_validation(cfg)
    train_loader = dataloader_builder(
        cfg,
        split="train",
        sample_n=train_n,
        dataframe=train_frame,
    )
    validation_loader = dataloader_builder(
        cfg,
        split="validation",
        sample_n=eval_n,
        dataframe=validation_frame,
    )
    return train_loader, validation_loader


def get_tree_position_train_validation_dataloaders(
    cfg,
    train_n=None,
    eval_n=None,
):
    return _build_train_validation_dataloaders(
        cfg,
        get_dataloader,
        train_n=train_n,
        eval_n=eval_n,
    )


def get_token_train_validation_dataloaders(
    cfg,
    train_n=None,
    eval_n=None,
):
    return _build_train_validation_dataloaders(
        cfg,
        get_token_dataloader,
        train_n=train_n,
        eval_n=eval_n,
    )


def get_tree_lstm_train_validation_dataloaders(
    cfg,
    train_n=None,
    eval_n=None,
):
    return _build_train_validation_dataloaders(
        cfg,
        get_tree_lstm_dataloader,
        train_n=train_n,
        eval_n=eval_n,
    )
