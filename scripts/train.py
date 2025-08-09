# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import load_config
from src.utils.io import load_vocab
from src.data.loader import get_dataloader
from src.data.dataset import load_split
from src.models.tree_transformer import TreeTransformer
from src.training.trainer import train
from src.training.evaluation import test_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train_config.yaml",
                   help="Path to config file.")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint to resume from (overrides config).")
    return p.parse_args()

def main():
    # --- args & config ---
    args = parse_args()  # NEW
    cfg = load_config(args.config)  # CHANGED: use CLI config path if provided

    # Resolve resume path: CLI > config > None
    resume_from = args.resume_from or getattr(getattr(cfg, "training", {}), "resume_from", None)  # NEW
    if resume_from and not os.path.exists(resume_from):
        print(f"[warn] resume_from does not exist: {resume_from} (ignoring)")
        resume_from = None

    # 1) Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) num_labels from train split
    df_train = load_split(cfg, split="train")
    num_labels = len(df_train['label'].iloc[0])
    print(f"Detected num_labels = {num_labels}")

    # 3) Dataloaders
    train_loader = get_dataloader(cfg, split="train", sample_n=10000)
    val_loader   = get_dataloader(cfg, split="test")

    # 4) Model
    vocab = load_vocab(cfg)
    model = TreeTransformer(
        vocab_size=len(vocab),
        d_model=cfg.model.d_model,
        nhead=cfg.model.heads,
        num_layers=cfg.model.layers,
        dim_feedforward=cfg.model.dim_feedforward,
        num_labels=num_labels,
        n=cfg.tree.branching_factor,
        k=cfg.tree.depth
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 5) Optimizer / Scheduler / Loss
    print("weight decay:", cfg.training.weight_decay)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=float(cfg.training.weight_decay)
    )
    total_steps = len(train_loader) * cfg.training.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    criterion = nn.MSELoss(reduction='none')

    # 6) Train (pass resume_from)
    trained_model = train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        resume_from=resume_from,  # NEW
    )

    # 7) Final evaluation
    print("\n== Final evaluation on test split ==")
    _, loss = test_model(trained_model, val_loader, device, criterion)
    print(f"Test total loss: {loss:.4f}")

if __name__ == "__main__":
    main()
