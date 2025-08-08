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


def train(
    cfg,
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion,
    device: torch.device
):
    """
    Full training loop over epochs with checkpointing and early stopping.
    """
    best_val = float("inf")
    epochs_no_improve = 0

    # Prepare checkpoint paths
    save_dir = cfg.paths.save_dir
    os.makedirs(save_dir, exist_ok=True)
    name = getattr(cfg, "experiment_name", "model")
    ckpt_path = os.path.join(save_dir, f"{name}.pth")
    best_path = os.path.join(save_dir, f"{name}_best.pth")

    for epoch in range(1, cfg.training.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.training.epochs} ===")

        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            log_interval=cfg.training.log_interval
        )
        print(f"[train] epoch {epoch} avg rank loss = {train_loss:.4f}")

        # Validation
        loss = eval_epoch(
            model, val_loader, device, criterion
        )
        print(f"[eval]  epoch {epoch} total loss = {loss:.4f}")

        # Scheduler step (works for Plateau or standard)
        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step()

        # Checkpoint & Early stopping
        is_best = False
        if val_total < best_val:
            best_val = val_total
            epochs_no_improve = 0
            is_best = True
        else:
            epochs_no_improve += 1

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val,
            "epochs_no_improve": epochs_no_improve
        }, ckpt_path)

        if is_best:
            torch.save(torch.load(ckpt_path), best_path)
            print(f"[checkpoint] New best model saved to {best_path}")

        # Early stop if no improvement
        patience = getattr(cfg.training, "early_stop_patience", 10)
        if epochs_no_improve >= patience:
            print(f"[early stop] stopping after {epochs_no_improve} epochs without improvement.")
            break

    return model
