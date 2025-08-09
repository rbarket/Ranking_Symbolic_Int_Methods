import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.evaluation import test_model

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int
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

    for step, (inputs, pos_enc, token_mask, labels, label_masks) in enumerate(loader, start=1):
        # Move tensors to device
        inputs     = inputs.to(device)
        pos_enc    = pos_enc.to(device)
        token_mask = token_mask.to(device)
        labels     = labels.to(device).float()
        label_masks= label_masks.to(device)

        optimizer.zero_grad()
        preds = model(inputs, pos_enc, token_mask, label_masks).squeeze()  # [B, L]

        # Compute true ranks
        y_inf = labels.clone()
        y_inf[~label_masks] = float('inf')
        true_ranks = torch.argsort(torch.argsort(y_inf, dim=1), dim=1).float()

        # Pairwise RankNet loss
        p1 = preds.unsqueeze(2); p2 = preds.unsqueeze(1)
        y1 = labels.unsqueeze(2); y2 = labels.unsqueeze(1)
        m1 = label_masks.unsqueeze(2); m2 = label_masks.unsqueeze(1)
        order_mask = m1 & m2 & (y1 < y2)
        pair_losses = F.softplus(p1 - p2) * order_mask.float()

        # Rank‐based weights
        r1 = true_ranks.unsqueeze(2); r2 = true_ranks.unsqueeze(1)
        avg_rank = (r1 + r2) / 2.0
        rank_weights = 1.0 / torch.log2(avg_rank + 2.0)
        weighted = pair_losses * (rank_weights * order_mask.float())

        total_pair_loss = weighted.sum()
        sum_weights     = (rank_weights * order_mask.float()).sum().clamp(min=1.0)
        rank_loss       = total_pair_loss / sum_weights

        # Backpropagation
        rank_loss.backward()
        optimizer.step()

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
    criterion
):
    """
    Runs evaluation using test_model from evaluation.py.
    Returns:
      - total_loss:    combined loss value
      - avg_rank_loss: average rank loss
    """
    _, loss = test_model(model, loader, device, criterion)
    return loss

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, epochs_no_improve):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on most OSes

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_loss = ckpt.get("best_val_loss", float("inf"))
    epochs_no_improve = ckpt.get("epochs_no_improve", 0)
    return start_epoch, best_loss, epochs_no_improve

def train(
    cfg,
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion,
    device: torch.device,
    resume_from: str | None = None,
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

    # defaults
    best_loss = float("inf")
    epochs_no_improve = 0
    start_epoch = 1

    # optional resume
    if resume_from is not None and os.path.exists(resume_from):
        start_epoch, best_loss, epochs_no_improve = load_checkpoint(
            resume_from, model, optimizer, scheduler, map_location=device
        )
        print(f"[resume] loaded {resume_from} -> start_epoch={start_epoch}, best_loss={best_loss:.4f}")
    else:
        print("no checkpoint found, starting from scratch")

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.training.epochs} ===")

        # --- Train ---
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            log_interval=cfg.training.log_interval
        )
        print(f"[train] epoch {epoch} avg loss = {train_loss:.4f}")

        # --- Validate ---
        val_loss = eval_epoch(model, val_loader, device, criterion)
        print(f"[eval]  epoch {epoch} val loss = {val_loss:.4f}")

        # --- Scheduler step ---
        scheduler.step()

        # --- Track best / early stopping ---
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # --- Save "last" ---
        save_checkpoint(
            ckpt_path, model, optimizer, scheduler,
            epoch=epoch, best_val_loss=best_loss, epochs_no_improve=epochs_no_improve
        )

        # --- Save "best" ---
        if is_best:
            # copy last to best (device-agnostic)
            torch.save(torch.load(ckpt_path, map_location="cpu"), best_path)
            print(f"[checkpoint] new best -> {best_path}")

        # --- Early stop ---
        patience = getattr(cfg.training, "early_stop_patience", 10)
        if epochs_no_improve >= patience:
            print(f"[early stop] stopping after {epochs_no_improve} epochs without improvement.")
            break

    return model

