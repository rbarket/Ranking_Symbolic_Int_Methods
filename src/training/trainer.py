from __future__ import annotations

import csv
import os
import time
import torch
import torch.nn as nn

from src.training.evaluation import ranknet_loss_components, test_model
from src.utils.checkpoint import (
    checkpoint_state_dict,
    is_v2_checkpoint,
    load_checkpoint_payload,
    restore_rng_state,
    save_checkpoint,
    save_checkpoint_payload,
    validate_checkpoint_hyperparameters,
    validate_checkpoint_model_type,
    validate_embedded_vocabulary,
    vocabulary_sha256,
)
from src.utils.config import config_to_dict

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
    batch_forward,
    scheduler=None,
) -> float:
    """
    Runs one epoch of training with the pairwise RankNet loss.
    Logs average loss every log_interval steps.
    Returns:
        epoch_avg_loss: average rank loss over the epoch
    """
    model.train()
    running_sum = 0.0
    running_weight = 0.0
    window_sum = 0.0
    window_weight = 0.0

    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        result = batch_forward(
            model,
            batch,
            device,
            apply_label_mask=True,
        )
        preds = result.predictions
        labels = result.labels
        label_masks = result.label_mask

        total_pair_loss, sum_weights, rank_loss = ranknet_loss_components(
            preds, labels, label_masks
        )

        # Backpropagation
        rank_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Accumulate for metrics
        running_sum    += total_pair_loss.item()
        running_weight += sum_weights.item()
        window_sum     += total_pair_loss.item()
        window_weight  += sum_weights.item()

        # Log running window
        if step % log_interval == 0:
            window_avg = window_sum / (window_weight + 1e-8)
            print(f"[train] step {step}/{len(loader)}  avg rank loss {window_avg:.4f}")
            window_sum = 0.0
            window_weight = 0.0

    # Compute epoch average
    epoch_avg_loss = running_sum / (running_weight + 1e-8)
    return epoch_avg_loss


def eval_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion,
    batch_forward,
):
    """
    Runs evaluation using test_model from evaluation.py.
    Returns:
      - total_loss:    combined loss value
      - avg_rank_loss: average rank loss
    """
    _, _, metrics = test_model(
        model,
        loader,
        device,
        criterion,
        batch_forward,
        return_metrics=True,
    )
    return metrics


_METRIC_FIELDS = (
    "epoch",
    "train_rank_loss",
    "validation_rank_loss",
    "validation_masked_mse",
    "validation_smallest_label_accuracy",
    "epoch_seconds",
    "learning_rate",
    "is_best",
)


def _prepare_metrics_file(path: str, resume_epoch: int | None = None) -> None:
    if resume_epoch is not None and os.path.exists(path):
        with open(path, newline="") as file:
            rows = list(csv.DictReader(file))
        epochs = [int(row["epoch"]) for row in rows]
        if epochs and epochs[-1] < resume_epoch:
            raise ValueError(
                f"Metrics file ends at epoch {epochs[-1]}, before checkpoint epoch "
                f"{resume_epoch}. Refusing to append an inconsistent history."
            )
        retained = [row for row in rows if int(row["epoch"]) <= resume_epoch]
        if len(retained) != len(rows):
            with open(path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=_METRIC_FIELDS)
                writer.writeheader()
                writer.writerows(retained)
            print(f"[resume] truncated metrics history after epoch {resume_epoch}")
        return
    with open(path, "w", newline="") as file:
        csv.DictWriter(file, fieldnames=_METRIC_FIELDS).writeheader()


def _append_epoch_metrics(path: str, row: dict) -> None:
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_METRIC_FIELDS)
        writer.writerow(row)


def _checkpoint_config_snapshot(cfg) -> dict:
    snapshot = config_to_dict(cfg)
    # The vocabulary has its own validated checkpoint field; avoid storing a
    # second copy after checkpoint-only resume injected it into the runtime cfg.
    snapshot.get("data", {}).pop("embedded_vocab", None)
    return snapshot

def _validate_resume_data(checkpoint: dict, current: dict) -> None:
    saved = checkpoint.get("data_metadata", {})
    immutable = (
        "batch_size",
        "training_n",
        "evaluation_n",
        "validation_fraction",
        "split_seed",
        "steps_per_epoch",
        "train_rows",
        "validation_rows",
        "train_data_sha256",
        "validation_data_sha256",
    )
    differences = [
        key for key in immutable
        if key in saved and saved.get(key) != current.get(key)
    ]
    if differences:
        raise ValueError(
            "Resume data settings differ for: "
            f"{', '.join(differences)}. Use --init_from for changed training data."
        )


def _resume_training_state(
    checkpoint,
    model,
    optimizer,
    scheduler,
    cfg,
    model_spec,
    model_hyperparameters,
    vocabulary,
    data_metadata,
    steps_per_epoch,
    device,
):
    validate_checkpoint_model_type(checkpoint, model_spec.model_type)
    validate_checkpoint_hyperparameters(checkpoint, model_hyperparameters)
    completed_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", completed_epoch * steps_per_epoch))
    scheduler_start = int(checkpoint.get("scheduler_start_global_step", 0))
    requested_epochs = int(cfg.training.epochs)
    if requested_epochs < completed_epoch:
        raise ValueError(
            f"Requested total epochs ({requested_epochs}) is before checkpoint epoch "
            f"{completed_epoch}. Choose at least {completed_epoch}."
        )

    if is_v2_checkpoint(checkpoint):
        embedded_vocab = validate_embedded_vocabulary(checkpoint)
        if vocabulary_sha256(embedded_vocab) != vocabulary_sha256(vocabulary):
            raise ValueError(
                "Resume vocabulary differs from the embedded checkpoint vocabulary. "
                "Use --init_from for a changed vocabulary."
            )
        _validate_resume_data(checkpoint, data_metadata)

    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint_state_dict(checkpoint))
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None:
        raise ValueError("Resume checkpoint does not contain optimizer state; use --init_from.")
    optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    planned_epochs = int(checkpoint.get("planned_epochs", requested_epochs))
    scheduler_total = int((scheduler_state or {}).get("total_steps", 0))
    scheduler_end = scheduler_start + scheduler_total
    should_extend = requested_epochs > planned_epochs or global_step >= scheduler_end
    remaining_updates = max(0, (requested_epochs - completed_epoch) * steps_per_epoch)

    if scheduler is not None and scheduler_state is not None and not should_extend:
        scheduler.load_state_dict(scheduler_state)
        print(
            f"[resume] restored OneCycleLR at optimizer step {global_step}/"
            f"{scheduler_end}"
        )
    elif remaining_updates > 0:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg.training.learning_rate),
            total_steps=remaining_updates,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        scheduler_start = global_step
        print(
            "[resume] requested training extends the saved OneCycle schedule; "
            f"started a new schedule for {remaining_updates} remaining updates"
        )
    else:
        scheduler = None

    restore_rng_state(checkpoint.get("rng_state"))
    return (
        completed_epoch + 1,
        float(checkpoint.get("best_val_loss", float("inf"))),
        int(checkpoint.get("epochs_no_improve", 0)),
        global_step,
        scheduler,
        scheduler_start,
    )

def train(
    cfg,
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion,
    device: torch.device,
    model_spec,
    model_hyperparameters,
    resume_from: str | None = None,
    vocabulary: dict | None = None,
    data_metadata: dict | None = None,
    resume_checkpoint: dict | None = None,
):
    """
    Full training loop over epochs with checkpointing and early stopping.
    """
    # checkpoint paths
    save_dir = cfg.paths.save_dir
    os.makedirs(save_dir, exist_ok=True)
    name = getattr(cfg, "experiment_name", "model")
    ckpt_path = os.path.join(save_dir, f"{name}.pth")
    best_path = os.path.join(save_dir, f"{name}_best.pth")
    metrics_path = os.path.join(save_dir, f"{name}_metrics.csv")

    # defaults
    best_loss = float("inf")
    epochs_no_improve = 0
    start_epoch = 1
    global_step = 0
    scheduler_start_global_step = 0
    steps_per_epoch = len(train_loader)
    vocabulary = vocabulary or {}
    data_metadata = dict(data_metadata or {})
    data_metadata.setdefault("steps_per_epoch", steps_per_epoch)

    # optional resume
    if resume_from is not None:
        checkpoint = resume_checkpoint or load_checkpoint_payload(
            resume_from,
            map_location=device,
        )
        if checkpoint.get("checkpoint_role") == "best":
            print(
                "[resume] warning: this is a best checkpoint; the last checkpoint "
                "is preferred for exact continuation"
            )
        (
            start_epoch,
            best_loss,
            epochs_no_improve,
            global_step,
            scheduler,
            scheduler_start_global_step,
        ) = _resume_training_state(
            checkpoint,
            model,
            optimizer,
            scheduler,
            cfg,
            model_spec,
            model_hyperparameters,
            vocabulary,
            data_metadata,
            steps_per_epoch,
            device,
        )
        print(f"[resume] loaded {resume_from} -> start_epoch={start_epoch}, best_loss={best_loss:.4f}")
    else:
        print("[train] starting from scratch")

    _prepare_metrics_file(
        metrics_path,
        resume_epoch=(start_epoch - 1 if start_epoch > 1 else None),
    )

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.training.epochs} ===")
        epoch_started = time.perf_counter()

        # --- Train ---
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            log_interval=cfg.training.log_interval,
            batch_forward=model_spec.forward_batch,
            scheduler=scheduler,
        )
        global_step += steps_per_epoch
        print(f"[train] epoch {epoch} avg loss = {train_loss:.4f}")

        # --- Validate ---
        val_metrics = eval_epoch(
            model,
            val_loader,
            device,
            criterion,
            model_spec.forward_batch,
        )
        val_loss = val_metrics.rank_loss
        print(f"[eval]  epoch {epoch} val loss = {val_loss:.4f}")

        # --- Track best / early stopping ---
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        _append_epoch_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "train_rank_loss": train_loss,
                "validation_rank_loss": val_metrics.rank_loss,
                "validation_masked_mse": val_metrics.masked_mse,
                "validation_smallest_label_accuracy": (
                    val_metrics.smallest_label_accuracy
                ),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": is_best,
            },
        )

        # --- Save "last" ---
        save_checkpoint(
            ckpt_path, model, optimizer, scheduler,
            epoch=epoch,
            best_val_loss=best_loss,
            epochs_no_improve=epochs_no_improve,
            model_type=model_spec.model_type,
            model_hyperparameters=model_hyperparameters,
            vocabulary=vocabulary,
            config_snapshot=_checkpoint_config_snapshot(cfg),
            data_metadata=data_metadata,
            checkpoint_role="last",
            global_step=global_step,
            planned_epochs=int(cfg.training.epochs),
            steps_per_epoch=steps_per_epoch,
            scheduler_start_global_step=scheduler_start_global_step,
        )

        # --- Save "best" ---
        if is_best:
            best_payload = load_checkpoint_payload(ckpt_path, map_location="cpu")
            best_payload["checkpoint_role"] = "best"
            save_checkpoint_payload(best_path, best_payload)
            print(f"[checkpoint] new best -> {best_path}")

        # --- Early stop ---
        patience = getattr(cfg.training, "early_stop_patience", 10)
        if epochs_no_improve >= patience:
            print(f"[early stop] stopping after {epochs_no_improve} epochs without improvement.")
            break

    if os.path.exists(best_path):
        best_checkpoint = load_checkpoint_payload(best_path, map_location=device)
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(checkpoint_state_dict(best_checkpoint))
        print(f"[checkpoint] restored best model from {best_path}")

    return model
