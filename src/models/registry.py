from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

import torch
import torch.nn as nn

from src.data.loader import (
    get_dataloader,
    get_tree_lstm_dataloader,
    get_tree_lstm_train_validation_dataloaders,
    get_token_dataloader,
    get_token_train_validation_dataloaders,
    get_tree_position_train_validation_dataloaders,
)
from src.models.lstm import LSTM
from src.models.transformer import Transformer
from src.models.tree_lstm import TreeLSTM
from src.models.tree_transformer import TreeTransformer


@dataclass(frozen=True)
class BatchResult:
    """Model predictions and their corresponding supervised targets."""

    predictions: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor


BatchAdapter = Callable[
    [nn.Module, Any, torch.device, bool],
    BatchResult,
]
HyperparameterResolver = Callable[[Any, int, int], dict[str, Any]]


@dataclass(frozen=True)
class ModelSpec:
    """Everything an entrypoint needs to run one model architecture."""

    model_type: str
    model_class: type[nn.Module]
    hyperparameter_resolver: HyperparameterResolver
    batch_adapter: BatchAdapter
    dataloader_builder: Callable[..., Any] = get_dataloader
    train_validation_dataloader_builder: Callable[..., Any] = (
        get_tree_position_train_validation_dataloaders
    )
    supports_data_parallel: bool = True
    dependency_check: Callable[[], None] | None = None

    def validate_environment(self) -> None:
        if self.dependency_check is not None:
            self.dependency_check()

    def validate_data_parallel(self, requested: bool) -> None:
        if requested and not self.supports_data_parallel:
            raise ValueError(
                f"model.type='{self.model_type}' does not support the current "
                "PyTorch DataParallel path. Run it without --data_parallel."
            )

    def resolve_hyperparameters(
        self,
        cfg,
        vocab_size: int,
        num_labels: int,
    ) -> dict[str, Any]:
        return self.hyperparameter_resolver(cfg, vocab_size, num_labels)

    def build_model(self, cfg, vocab_size: int, num_labels: int) -> nn.Module:
        return self.model_class(
            **self.resolve_hyperparameters(cfg, vocab_size, num_labels)
        )

    def build_model_from_hyperparameters(
        self,
        hyperparameters: dict[str, Any],
    ) -> nn.Module:
        """Construct a model from the resolved parameters embedded in a checkpoint."""
        return self.model_class(**dict(hyperparameters))

    def build_dataloader(self, cfg, split="train", sample_n=None, shuffle=None):
        return self.dataloader_builder(
            cfg,
            split=split,
            sample_n=sample_n,
            shuffle=shuffle,
        )

    def build_train_validation_dataloaders(
        self,
        cfg,
        train_n=None,
        eval_n=None,
    ):
        return self.train_validation_dataloader_builder(
            cfg,
            train_n=train_n,
            eval_n=eval_n,
        )

    def forward_batch(
        self,
        model: nn.Module,
        batch,
        device: torch.device,
        apply_label_mask: bool = False,
    ) -> BatchResult:
        return self.batch_adapter(model, batch, device, apply_label_mask)


def _validate_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if predictions.shape != labels.shape:
        raise ValueError(
            "Model predictions and labels must have the same [batch, labels] "
            f"shape, got {tuple(predictions.shape)} and {tuple(labels.shape)}."
        )


def tree_position_batch_adapter(
    model: nn.Module,
    batch,
    device: torch.device,
    apply_label_mask: bool = False,
) -> BatchResult:
    """Run a tree-position sequence batch through TreeTransformer."""
    if len(batch) != 5:
        raise ValueError(
            "Sequence models expect batches containing token IDs, positional "
            "encodings, token masks, labels, and label masks."
        )

    token_ids, pos_encodings, token_mask, labels, label_mask = batch
    token_ids = token_ids.to(device)
    pos_encodings = pos_encodings.to(device)
    token_mask = token_mask.to(device)
    labels = labels.to(device).float()
    label_mask = label_mask.to(device)

    predictions = model(
        token_ids,
        pos_encodings,
        token_mask,
        label_mask if apply_label_mask else None,
    )
    _validate_predictions(predictions, labels)

    return BatchResult(predictions, labels, label_mask)


def _prepare_token_batch(batch, device: torch.device):
    if len(batch) != 4:
        raise ValueError(
            "Token sequence models expect batches containing token IDs, token "
            "masks, labels, and label masks."
        )

    token_ids, token_mask, labels, label_mask = batch
    return (
        token_ids.to(device),
        token_mask.to(device),
        labels.to(device).float(),
        label_mask.to(device),
    )


def transformer_batch_adapter(
    model: nn.Module,
    batch,
    device: torch.device,
    apply_label_mask: bool = False,
) -> BatchResult:
    token_ids, token_mask, labels, label_mask = _prepare_token_batch(batch, device)
    predictions = model(
        token_ids,
        None,
        token_mask,
        label_mask if apply_label_mask else None,
    )
    _validate_predictions(predictions, labels)

    return BatchResult(predictions, labels, label_mask)


def lstm_batch_adapter(
    model: nn.Module,
    batch,
    device: torch.device,
    apply_label_mask: bool = False,
) -> BatchResult:
    token_ids, token_mask, labels, label_mask = _prepare_token_batch(batch, device)
    predictions = model(
        token_ids,
        token_mask,
        label_mask if apply_label_mask else None,
    )
    _validate_predictions(predictions, labels)

    return BatchResult(predictions, labels, label_mask)


def tree_lstm_batch_adapter(
    model: nn.Module,
    batch,
    device: torch.device,
    apply_label_mask: bool = False,
) -> BatchResult:
    if len(batch) != 3:
        raise ValueError(
            "TreeLSTM expects batches containing a DGL graph, labels, and "
            "label masks."
        )

    graph, labels, label_mask = batch
    graph = graph.to(device)
    labels = labels.to(device).float()
    label_mask = label_mask.to(device)
    predictions = model(
        graph,
        label_mask if apply_label_mask else None,
    )
    _validate_predictions(predictions, labels)
    return BatchResult(predictions, labels, label_mask)


def _tree_transformer_hyperparameters(
    cfg,
    vocab_size: int,
    num_labels: int,
) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "d_model": int(cfg.model.d_model),
        "nhead": int(cfg.model.heads),
        "num_layers": int(cfg.model.layers),
        "dim_feedforward": int(cfg.model.dim_feedforward),
        "num_labels": num_labels,
        "n": int(cfg.tree.branching_factor),
        "k": int(cfg.tree.depth),
        "dropout": float(getattr(cfg.model, "dropout", 0.1)),
        "activation": str(getattr(cfg.model, "activation", "gelu")),
    }


def _transformer_hyperparameters(
    cfg,
    vocab_size: int,
    num_labels: int,
) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "d_model": int(cfg.model.d_model),
        "nhead": int(cfg.model.heads),
        "num_layers": int(cfg.model.layers),
        "dim_feedforward": int(cfg.model.dim_feedforward),
        "num_labels": num_labels,
        "max_seq_len": int(getattr(cfg.model, "max_seq_len", 1024)),
        "dropout": float(getattr(cfg.model, "dropout", 0.1)),
        "activation": str(getattr(cfg.model, "activation", "gelu")),
    }


def _lstm_hyperparameters(
    cfg,
    vocab_size: int,
    num_labels: int,
) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "d_model": int(cfg.model.d_model),
        "num_layers": int(cfg.model.layers),
        "num_labels": num_labels,
        "dropout": float(getattr(cfg.model, "dropout", 0.1)),
    }


def _tree_lstm_hyperparameters(
    cfg,
    vocab_size: int,
    num_labels: int,
) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "d_model": int(cfg.model.d_model),
        "hidden_size": int(cfg.model.hidden_size),
        "num_labels": num_labels,
        "dropout": float(getattr(cfg.model, "dropout", 0.1)),
    }


def _require_dgl() -> None:
    try:
        import_module("dgl")
    except ImportError as exc:
        raise RuntimeError(
            "model.type='tree_lstm' requires DGL. Run this command in the "
            "TreeLSTM_DGL environment."
        ) from exc


_MODEL_REGISTRY = {
    "tree_transformer": ModelSpec(
        model_type="tree_transformer",
        model_class=TreeTransformer,
        hyperparameter_resolver=_tree_transformer_hyperparameters,
        batch_adapter=tree_position_batch_adapter,
        dataloader_builder=get_dataloader,
        train_validation_dataloader_builder=(
            get_tree_position_train_validation_dataloaders
        ),
    ),
    "transformer": ModelSpec(
        model_type="transformer",
        model_class=Transformer,
        hyperparameter_resolver=_transformer_hyperparameters,
        batch_adapter=transformer_batch_adapter,
        dataloader_builder=get_token_dataloader,
        train_validation_dataloader_builder=(
            get_token_train_validation_dataloaders
        ),
    ),
    "lstm": ModelSpec(
        model_type="lstm",
        model_class=LSTM,
        hyperparameter_resolver=_lstm_hyperparameters,
        batch_adapter=lstm_batch_adapter,
        dataloader_builder=get_token_dataloader,
        train_validation_dataloader_builder=(
            get_token_train_validation_dataloaders
        ),
    ),
    "tree_lstm": ModelSpec(
        model_type="tree_lstm",
        model_class=TreeLSTM,
        hyperparameter_resolver=_tree_lstm_hyperparameters,
        batch_adapter=tree_lstm_batch_adapter,
        dataloader_builder=get_tree_lstm_dataloader,
        train_validation_dataloader_builder=(
            get_tree_lstm_train_validation_dataloaders
        ),
        supports_data_parallel=False,
        dependency_check=_require_dgl,
    ),
}


def available_model_types() -> tuple[str, ...]:
    return tuple(_MODEL_REGISTRY)


def resolve_model_type(cfg) -> str:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        raise ValueError("Configuration must contain a model section.")

    model_type = str(getattr(model_cfg, "type", "tree_transformer")).strip().lower()
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model.type='{model_type}'. Available model types: "
            f"{', '.join(available_model_types())}."
        )
    return model_type


def get_model_spec(cfg) -> ModelSpec:
    return get_model_spec_by_type(resolve_model_type(cfg))


def get_model_spec_by_type(model_type: str) -> ModelSpec:
    normalized = str(model_type).strip().lower()
    if normalized not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model.type='{normalized}'. Available model types: "
            f"{', '.join(available_model_types())}."
        )
    spec = _MODEL_REGISTRY[normalized]
    spec.validate_environment()
    return spec
