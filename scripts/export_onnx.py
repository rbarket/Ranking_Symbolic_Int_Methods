#!/usr/bin/env python3
"""
Export a trained TreeTransformer checkpoint to ONNX for use in C runtimes.

Usage:
  python scripts/export_onnx.py \
      --config configs/train_config.yaml \
      --checkpoint models/ranking/model_best.pth \
      --output models/ranking/tree_transformer.onnx
"""

import argparse
import os
import torch

from src.utils.config import load_config
from src.utils.io import load_vocab
from src.data.dataset import PrefixExpressionDataset, collate_fn
from src.models.tree_transformer import TreeTransformer


def resolve_device(device_arg: str) -> torch.device:
    """
    device_arg: 'auto' | 'cpu' | 'cuda' | 'cuda:0'...
    """
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        return torch.device(device_arg if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Export TreeTransformer to ONNX.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to the training config used for this checkpoint.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pth checkpoint. If omitted, tries <save_dir>/<name>_best.pth then <name>.pth",
    )
    p.add_argument(
        "--output",
        type=str,
        default="models/ranking/tree_transformer.onnx",
        help="Where to write the ONNX file.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export: auto | cpu | cuda | cuda:{idx}",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset to target.",
    )
    p.add_argument(
        "--dynamo",
        dest="dynamo",
        action="store_true",
        default=True,
        help="Use the dynamo exporter for better dynamic shape support (default: on).",
    )
    p.add_argument(
        "--no-dynamo",
        dest="dynamo",
        action="store_false",
        help="Disable the dynamo exporter and use the legacy exporter.",
    )
    p.add_argument(
        "--num_labels",
        type=int,
        default=None,
        help="Skip dataset load and force a specific num_labels for the head.",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Skip dataset load and use a dummy sequence length (default 4) for export.",
    )
    return p.parse_args()


def find_checkpoint(cfg, override: str | None) -> str:
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"Checkpoint not found: {override}")
        return override

    name = getattr(cfg, "experiment_name", "model")
    ranking_best = os.path.join(cfg.paths.save_dir, "ranking_best.pth")
    best = os.path.join(cfg.paths.save_dir, f"{name}_best.pth")
    last = os.path.join(cfg.paths.save_dir, f"{name}.pth")
    for cand in (ranking_best, best, last):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"No checkpoint found. Looked for: {best} and {last}. "
        "Pass --checkpoint to specify one explicitly."
    )


def load_sample(cfg, num_labels_override: int | None, seq_len_override: int | None):
    """
    Build a sample batch for tracing. If overrides are provided, use a lightweight
    synthetic example to avoid loading the full dataset.
    """
    if num_labels_override is not None or seq_len_override is not None:
        seq_len = seq_len_override if seq_len_override is not None else 4
        num_labels = num_labels_override if num_labels_override is not None else 1
        token_ids = torch.ones((1, seq_len), dtype=torch.long)
        pos_enc = torch.zeros((1, seq_len, cfg.model.d_model), dtype=torch.float32)
        token_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        print(f"[export] using synthetic sample: seq_len={seq_len}, num_labels={num_labels}")
        return token_ids, pos_enc, token_mask, num_labels

    # Fallback: grab a real example from the dataset
    print("[export] loading train split to grab a sample...")
    ds = PrefixExpressionDataset(cfg, split="test")
    if len(ds) == 0:
        raise RuntimeError("Training split is empty; cannot build example input.")
    sample = ds[0]
    print(f"[export] dataset size: {len(ds)} (using first example)")
    token_ids, pos_enc, token_mask, labels, label_masks = collate_fn([sample])
    num_labels = labels.shape[1]
    return token_ids, pos_enc, token_mask, num_labels


def build_model(cfg, vocab_size: int, num_labels: int, device: torch.device) -> TreeTransformer:
    model = TreeTransformer(
        vocab_size=vocab_size,
        d_model=cfg.model.d_model,
        nhead=cfg.model.heads,
        num_layers=cfg.model.layers,
        dim_feedforward=cfg.model.dim_feedforward,
        num_labels=num_labels,
        n=cfg.tree.branching_factor,
        k=cfg.tree.depth,
    )
    return model.to(device)


def load_weights(model: TreeTransformer, checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    # Handle DataParallel checkpoints that prefix keys with "module."
    if isinstance(state_dict, dict) and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)

    ckpt_path = find_checkpoint(cfg, args.checkpoint)
    print(f"[export] using checkpoint: {ckpt_path}")

    vocab = load_vocab(cfg)
    vocab_size = len(vocab)
    print(f"[export] vocab size: {vocab_size}")

    token_ids, pos_enc, token_mask, num_labels = load_sample(
        cfg, args.num_labels, args.seq_len
    )
    print(
        f"[export] sample shapes: token_ids {tuple(token_ids.shape)}, "
        f"pos_enc {tuple(pos_enc.shape)}, token_mask {tuple(token_mask.shape)}, "
        f"num_labels={num_labels}"
    )
    model = build_model(cfg, vocab_size, num_labels, device)
    print(f"[export] model built on device: {device}")
    load_weights(model, ckpt_path, device)
    print("[export] weights loaded")
    model.eval()

    token_ids = token_ids.to(device)
    pos_enc = pos_enc.to(device)
    token_mask = token_mask.to(device)
    print("[export] example batch moved to device")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dynamic_axes = None
    dynamic_shapes = None
    if args.dynamo:
        batch_dim = torch.export.Dim("batch")
        seq_dim = torch.export.Dim("seq")
        dynamic_shapes = (
            {0: batch_dim, 1: seq_dim},  # token_ids
            {0: batch_dim, 1: seq_dim},  # pos_encodings
            {0: batch_dim, 1: seq_dim},  # token_mask
        )
    else:
        dynamic_axes = {
            "token_ids": {0: "batch", 1: "seq"},
            "pos_encodings": {0: "batch", 1: "seq"},
            "token_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        }

    print(f"[export] exporting to {args.output} (opset {args.opset}) on device {device}")

    # Disable Transformer fastpath to avoid the fused _transformer_encoder_layer_fwd op
    # that lacks ONNX support.
    prev_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (token_ids, pos_enc, token_mask),
                args.output,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=["token_ids", "pos_encodings", "token_mask"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                dynamic_shapes=dynamic_shapes,
                dynamo=args.dynamo,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(prev_fastpath)
    print("[export] done.")


if __name__ == "__main__":
    main()
