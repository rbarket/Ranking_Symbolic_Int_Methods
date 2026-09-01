from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.data.dataset import load_split
from src.models.registry import get_model_spec
from src.training.evaluation import (
    EvaluationResult,
    compute_evaluation_metrics,
    evaluate_model,
)
from src.utils.checkpoint import (
    checkpoint_state_dict,
    is_v2_checkpoint,
    load_checkpoint_payload,
    reconstruct_model_from_checkpoint,
    validate_checkpoint_hyperparameters,
    validate_checkpoint_model_type,
    validate_embedded_vocabulary,
    validate_vocab_checkpoint_compatible,
    vocabulary_sha256,
)
from src.utils.config import config_from_dict, load_config, resolve_device
from src.utils.io import load_vocab, load_vocab_path
from src.utils.tree_utils import prefix_to_tree


def _resolve_context(checkpoint, config_path, input_dir, vocab_path):
    if config_path is not None:
        cfg = load_config(config_path)
    elif is_v2_checkpoint(checkpoint):
        cfg = config_from_dict(checkpoint["config"])
    else:
        raise ValueError("Legacy checkpoints require a matching --config file.")

    if input_dir is not None:
        cfg.data.input_dir = input_dir
    if vocab_path is not None:
        cfg.data.vocab_path = vocab_path

    if is_v2_checkpoint(checkpoint):
        embedded_vocab = validate_embedded_vocabulary(checkpoint)
        if vocab_path is not None:
            override_vocab = load_vocab_path(vocab_path)
            if vocabulary_sha256(override_vocab) != vocabulary_sha256(embedded_vocab):
                raise ValueError(
                    "--vocab_path does not match the vocabulary embedded in the checkpoint."
                )
        cfg.data.embedded_vocab = embedded_vocab
        vocab = embedded_vocab
        checkpoint_spec, model = reconstruct_model_from_checkpoint(checkpoint)
        if config_path is not None:
            config_spec = get_model_spec(cfg)
            validate_checkpoint_model_type(checkpoint, config_spec.model_type)
            num_labels = int(checkpoint["model_hyperparameters"]["num_labels"])
            resolved = config_spec.resolve_hyperparameters(cfg, len(vocab), num_labels)
            validate_checkpoint_hyperparameters(checkpoint, resolved)
        return cfg, checkpoint_spec, model, vocab

    spec = get_model_spec(cfg)
    validate_checkpoint_model_type(checkpoint, spec.model_type)
    vocab = load_vocab(cfg)
    split_frame = load_split(cfg, split="test")
    if split_frame.empty:
        raise ValueError("Cannot infer num_labels from an empty dataset.")
    num_labels = len(split_frame["label"].iloc[0])
    model = spec.build_model(cfg, len(vocab), num_labels)
    return cfg, spec, model, vocab


def _tree_depth(tokens) -> int:
    root = prefix_to_tree(tokens)

    def depth(node):
        if node is None:
            return -1
        return 1 + max(depth(node.left), depth(node.right))

    return depth(root)


def _true_minimum_indices(labels, mask):
    valid = [index for index, is_valid in enumerate(mask) if is_valid]
    if not valid:
        return []
    minimum = min(labels[index] for index in valid)
    return [index for index in valid if labels[index] == minimum]


def _prediction_frame(selected_frame: pd.DataFrame, result: EvaluationResult) -> pd.DataFrame:
    if len(selected_frame) != len(result.predictions):
        raise ValueError(
            "Dataset row metadata and predictions are misaligned: "
            f"{len(selected_frame)} rows versus {len(result.predictions)} predictions."
        )
    frame = selected_frame.reset_index(drop=True).copy()
    prefixes = frame["prefix"].tolist()
    labels = result.labels.tolist()
    masks = result.label_masks.tolist()
    predictions = result.predictions.tolist()
    predicted_best = []
    true_minima = []
    for prediction, label, mask in zip(predictions, labels, masks):
        valid = [index for index, is_valid in enumerate(mask) if is_valid]
        predicted_best.append(
            min(valid, key=lambda index: prediction[index]) if valid else None
        )
        true_minima.append(_true_minimum_indices(label, mask))

    return pd.DataFrame({
        "row_id": frame["row_id"].astype("int64"),
        "source": frame.get("source", pd.Series([None] * len(frame))),
        "prefix": prefixes,
        "integrand": frame.get("integrand", pd.Series([None] * len(frame))),
        "labels": labels,
        "label_mask": masks,
        "predictions": predictions,
        "predicted_best_index": predicted_best,
        "true_minimum_indices": true_minima,
        "token_count": [len(tokens) for tokens in prefixes],
        "tree_depth": [_tree_depth(tokens) for tokens in prefixes],
    })


def _metric_row(group_type, group, indices, result):
    subset = torch.as_tensor(indices, dtype=torch.long)
    metrics = compute_evaluation_metrics(
        result.predictions[subset],
        result.labels[subset],
        result.label_masks[subset],
    )
    return {"group_type": group_type, "group": group, "rows": len(indices), **metrics.to_dict()}


def _group_metrics(frame: pd.DataFrame, result: EvaluationResult):
    source_rows = []
    for source, group in frame.groupby("source", dropna=False, sort=True):
        source_rows.append(_metric_row("source", str(source), group.index.tolist(), result))

    complexity_rows = []
    specifications = (
        ("token_count", [-1, 16, 32, 64, float("inf")], ["1-16", "17-32", "33-64", "65+"]),
        ("tree_depth", [-1, 5, 10, 15, float("inf")], ["0-5", "6-10", "11-15", "16+"]),
    )
    for column, bins, names in specifications:
        groups = pd.cut(frame[column], bins=bins, labels=names)
        for name in names:
            indices = frame.index[groups == name].tolist()
            if indices:
                complexity_rows.append(_metric_row(column, name, indices, result))
    return pd.DataFrame(source_rows), pd.DataFrame(complexity_rows)


def main(
    checkpoint_path: str,
    split: str = "test",
    sample_n: int | None = None,
    device_arg: str = "auto",
    data_parallel: bool = False,
    config_path: str | None = None,
    input_dir: str | None = None,
    vocab_path: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    output_dir: str | None = None,
) -> EvaluationResult:
    if sample_n is not None and sample_n < 1:
        raise ValueError("sample_n must be at least 1.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers is not None and num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    checkpoint = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    cfg, model_spec, model, vocab = _resolve_context(
        checkpoint, config_path, input_dir, vocab_path
    )
    model_spec.validate_data_parallel(data_parallel)
    if batch_size is not None:
        cfg.training.batch_size = batch_size
    if num_workers is not None:
        cfg.data.num_workers = num_workers

    device = resolve_device(device_arg)
    print(f"Selected model type: {model_spec.model_type}")
    print(f"Using device: {device}")
    loader = model_spec.build_dataloader(
        cfg, split=split, sample_n=sample_n, shuffle=False
    )
    selected_frame = loader.dataset.selected_frame.reset_index(drop=True)
    state_dict = checkpoint_state_dict(checkpoint)
    validate_vocab_checkpoint_compatible(state_dict, len(vocab))
    model.load_state_dict(state_dict)
    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    print(f"Loaded checkpoint from {checkpoint_path}")

    result = evaluate_model(model, loader, device, model_spec.forward_batch)
    print(f"Inference complete: {tuple(result.predictions.shape)}")
    prediction_frame = _prediction_frame(selected_frame, result)
    source_metrics, complexity_metrics = _group_metrics(prediction_frame, result)

    destination = Path(output_dir) if output_dir else Path(checkpoint_path).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)
    sample_name = f"n{sample_n}" if sample_n is not None else "all"
    base = f"{Path(checkpoint_path).stem}_{split}_{sample_name}"
    prediction_frame.to_parquet(destination / f"{base}_predictions.parquet", index=False)
    with open(destination / f"{base}_metrics.json", "w") as file:
        json.dump(result.metrics.to_dict(), file, indent=2)
    source_metrics.to_csv(destination / f"{base}_metrics_by_source.csv", index=False)
    complexity_metrics.to_csv(destination / f"{base}_metrics_by_complexity.csv", index=False)
    print(f"Saved structured inference outputs to {destination}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--config", default=None,
                        help="Required for legacy checkpoints; optional for version 2.")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--vocab_path", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    arguments = parser.parse_args()
    main(
        checkpoint_path=arguments.checkpoint_path,
        split=arguments.split,
        sample_n=arguments.sample_n,
        device_arg=arguments.device,
        data_parallel=arguments.data_parallel,
        config_path=arguments.config,
        input_dir=arguments.input_dir,
        vocab_path=arguments.vocab_path,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        output_dir=arguments.output_dir,
    )
