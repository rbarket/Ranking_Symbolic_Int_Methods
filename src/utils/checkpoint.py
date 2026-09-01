from __future__ import annotations

import hashlib
import json
import os
import random
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 2


def load_checkpoint_payload(path: str, map_location="cpu") -> dict:
    """Load any supported checkpoint and normalize a raw state dict."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a dictionary, got {type(payload).__name__}.")
    if "model_state_dict" not in payload:
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return {"model_state_dict": payload, "legacy_raw_state_dict": True}
        raise KeyError("Checkpoint does not contain model_state_dict.")
    return payload


def is_v2_checkpoint(checkpoint: dict) -> bool:
    return int(checkpoint.get("checkpoint_format_version", 0)) >= CHECKPOINT_FORMAT_VERSION


def checkpoint_state_dict(checkpoint: dict) -> dict:
    return maybe_strip_module_prefix(checkpoint["model_state_dict"])


def vocabulary_sha256(vocab: dict[str, int]) -> str:
    canonical = json.dumps(vocab, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_embedded_vocabulary(checkpoint: dict) -> dict[str, int]:
    vocab = checkpoint.get("vocabulary")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Version-2 checkpoint is missing its embedded vocabulary.")
    expected_hash = checkpoint.get("vocabulary_sha256")
    actual_hash = vocabulary_sha256(vocab)
    if expected_hash != actual_hash:
        raise ValueError(
            "Checkpoint vocabulary hash mismatch; the checkpoint metadata may be corrupted."
        )
    return dict(vocab)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_v2_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    epochs_no_improve: int,
    model_type: str,
    model_hyperparameters: dict,
    vocabulary: dict,
    config_snapshot: dict,
    data_metadata: dict,
    checkpoint_role: str,
    planned_epochs: int,
    steps_per_epoch: int,
    scheduler_start_global_step: int = 0,
) -> dict:
    if checkpoint_role not in {"last", "best"}:
        raise ValueError("checkpoint_role must be 'last' or 'best'.")
    target_model = model.module if hasattr(model, "module") else model
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_role": checkpoint_role,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_type": str(model_type),
        "model_hyperparameters": deepcopy(model_hyperparameters),
        "model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler_state,
        "scheduler_start_global_step": int(scheduler_start_global_step),
        "best_val_loss": float(best_val_loss),
        "epochs_no_improve": int(epochs_no_improve),
        "planned_epochs": int(planned_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "planned_optimizer_steps": int(planned_epochs) * int(steps_per_epoch),
        "vocabulary": deepcopy(vocabulary),
        "vocabulary_sha256": vocabulary_sha256(vocabulary),
        "config": deepcopy(config_snapshot),
        "data_metadata": deepcopy(data_metadata),
        "rng_state": capture_rng_state(),
    }


def save_checkpoint_payload(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    epochs_no_improve,
    model_type=None,
    model_hyperparameters=None,
    vocabulary=None,
    config_snapshot=None,
    data_metadata=None,
    checkpoint_role="last",
    global_step=None,
    planned_epochs=None,
    steps_per_epoch=None,
    scheduler_start_global_step=0,
):
    """Save either a self-describing v2 checkpoint or a legacy payload."""
    if vocabulary and config_snapshot is not None and data_metadata is not None:
        payload = build_v2_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step if global_step is not None else 0,
            best_val_loss=best_val_loss,
            epochs_no_improve=epochs_no_improve,
            model_type=model_type,
            model_hyperparameters=model_hyperparameters,
            vocabulary=vocabulary,
            config_snapshot=config_snapshot,
            data_metadata=data_metadata,
            checkpoint_role=checkpoint_role,
            planned_epochs=planned_epochs if planned_epochs is not None else epoch,
            steps_per_epoch=steps_per_epoch if steps_per_epoch is not None else 0,
            scheduler_start_global_step=scheduler_start_global_step,
        )
    else:
        target_model = model.module if hasattr(model, "module") else model
        payload = {
            "epoch": epoch,
            "model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        }
        if model_type is not None:
            payload["model_type"] = model_type
        if model_hyperparameters is not None:
            payload["model_hyperparameters"] = model_hyperparameters
    save_checkpoint_payload(path, payload)
    return payload


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    expected_model_type=None,
):
    """Compatibility loader for model and optional optimizer/scheduler state."""
    checkpoint = load_checkpoint_payload(path, map_location=map_location)
    if expected_model_type is not None:
        validate_checkpoint_model_type(checkpoint, expected_model_type)
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint_state_dict(checkpoint))
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        float(checkpoint.get("best_val_loss", float("inf"))),
        int(checkpoint.get("epochs_no_improve", 0)),
    )


def validate_checkpoint_hyperparameters(
    checkpoint: dict,
    expected: dict,
) -> None:
    actual = checkpoint.get("model_hyperparameters")
    if actual is None:
        return
    if actual != expected:
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            "Checkpoint/model hyperparameter mismatch for: "
            f"{', '.join(differing)}. Use --init_from for intentional fine-tuning changes."
        )


def reconstruct_model_from_checkpoint(checkpoint: dict):
    """Build a v2 model through the central registry without a YAML file."""
    if not is_v2_checkpoint(checkpoint):
        raise ValueError("Legacy checkpoints require a matching --config file.")
    from src.models.registry import get_model_spec_by_type

    model_type = checkpoint.get("model_type")
    hyperparameters = checkpoint.get("model_hyperparameters")
    if not model_type or not isinstance(hyperparameters, dict):
        raise ValueError("Version-2 checkpoint is missing model construction metadata.")
    spec = get_model_spec_by_type(model_type)
    return spec, spec.build_model_from_hyperparameters(hyperparameters)


def validate_checkpoint_model_type(checkpoint: dict, expected_model_type: str) -> None:
    """Reject a typed checkpoint when it belongs to another architecture."""
    checkpoint_model_type = checkpoint.get("model_type")
    if checkpoint_model_type is None:
        return
    if checkpoint_model_type != expected_model_type:
        raise ValueError(
            "Checkpoint/model mismatch: checkpoint was saved for "
            f"model.type='{checkpoint_model_type}', but the supplied config selects "
            f"model.type='{expected_model_type}'."
        )


def maybe_strip_module_prefix(state_dict: dict) -> dict:
    """Strip DataParallel's `module.` prefix from checkpoint keys if present."""
    if not state_dict:
        return state_dict
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def get_embedding_weight(state_dict: dict) -> torch.Tensor:
    if "embedding.weight" not in state_dict:
        raise KeyError("Checkpoint state_dict does not contain embedding.weight")
    return state_dict["embedding.weight"]


def validate_vocab_checkpoint_compatible(state_dict: dict, vocab_size: int, *, allow_resize: bool = False) -> None:
    checkpoint_vocab_size = get_embedding_weight(state_dict).shape[0]
    if checkpoint_vocab_size == vocab_size:
        return
    if allow_resize and checkpoint_vocab_size < vocab_size:
        return
    raise ValueError(
        "Checkpoint/vocab mismatch: checkpoint embedding has "
        f"{checkpoint_vocab_size} rows but vocab has {vocab_size} tokens. "
        "Use the matching old vocab for old checkpoints, or use an explicit "
        "embedding-resize initialization path for fine-tuning."
    )


def _average_embedding_for_suffix(old_embedding: torch.Tensor, old_vocab: dict, suffix: str) -> Optional[torch.Tensor]:
    indices = [idx for token, idx in old_vocab.items() if token.endswith(suffix) and idx < old_embedding.shape[0]]
    if not indices:
        return None
    return old_embedding[torch.tensor(indices, dtype=torch.long)].mean(dim=0)


def resize_embedding_for_vocab(
    state_dict: dict,
    old_vocab: dict,
    new_vocab: dict,
    *,
    oov_token: str = "<OOV>",
) -> dict:
    """
    Return a copy of `state_dict` whose embedding matrix matches `new_vocab`.

    Existing token rows are copied by token id. New `_1`/`_2` function tokens are
    initialized from the average old unary/binary special-function embedding,
    falling back to `<OOV>` if no arity-matched average exists.
    """
    state_dict = dict(state_dict)
    old_embedding = get_embedding_weight(state_dict)
    if len(new_vocab) < old_embedding.shape[0]:
        raise ValueError(
            "Refusing to shrink checkpoint embeddings: checkpoint has "
            f"{old_embedding.shape[0]} rows but new vocab has {len(new_vocab)} tokens."
        )

    new_embedding = old_embedding.new_empty((len(new_vocab), old_embedding.shape[1]))
    oov_idx = old_vocab.get(oov_token, 1)
    fallback = old_embedding[oov_idx].clone()
    unary_average = _average_embedding_for_suffix(old_embedding, old_vocab, "_1")
    binary_average = _average_embedding_for_suffix(old_embedding, old_vocab, "_2")

    for token, new_idx in sorted(new_vocab.items(), key=lambda item: item[1]):
        old_idx = old_vocab.get(token)
        if old_idx is not None and old_idx < old_embedding.shape[0]:
            new_embedding[new_idx] = old_embedding[old_idx]
        elif token.endswith("_1") and unary_average is not None:
            new_embedding[new_idx] = unary_average
        elif token.endswith("_2") and binary_average is not None:
            new_embedding[new_idx] = binary_average
        else:
            new_embedding[new_idx] = fallback

    state_dict["embedding.weight"] = new_embedding
    return state_dict
