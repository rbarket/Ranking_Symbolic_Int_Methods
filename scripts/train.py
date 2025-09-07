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


def resolve_device(device_arg: str) -> torch.device:
    """
    device_arg: 'auto' | 'cpu' | 'cuda' | 'cuda:0'...
    Picks a sensible default and safely falls back to CPU if CUDA isn't available.
    """
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        return torch.device(device_arg if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train_config.yaml",
                   help="Path to config file.")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint to resume from (overrides config).")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | cuda:{idx}")
    p.add_argument("--data_parallel", action="store_true",
                   help="Use DataParallel when multiple GPUs are available (off by default).")
    p.add_argument("--n", type=int, default=None,
                   help="Number of training samples to use (overrides config.training.n if set).")
    return p.parse_args()


def main():
    # --- args & config ---
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve resume path: CLI > config > None
    resume_from = args.resume_from or getattr(getattr(cfg, "training", {}), "resume_from", None)
    if resume_from and not os.path.exists(resume_from):
        print(f"[warn] resume_from does not exist: {resume_from} (ignoring)")
        resume_from = None

    # 1) Device (explicit flag overrides anything in config)
    device = resolve_device(args.device)
    dp_ok = (args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1)
    picked = f"{device} (DataParallel x{torch.cuda.device_count()})" if dp_ok else f"{device}"
    print(f"Using device: {picked}")

    # 2) num_labels from train split
    df_train = load_split(cfg, split="train")
    num_labels = len(df_train['label'].iloc[0])
    print(f"Detected num_labels = {num_labels}")

    # 3) Dataloaders
    # (get_dataloader internally uses cfg; pin_memory/num_workers should be fine for CPU/GPU)
    sample_n = args.n if args.n is not None else getattr(cfg.training, "n", None)
    train_loader = get_dataloader(cfg, split="train", sample_n=sample_n)
    val_loader = get_dataloader(cfg, split="test") # TODO: need to divide out some data for val, instead of using test

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
    )

    # Optional multi-GPU for training
    if dp_ok:
        model = nn.DataParallel(model)

    model = model.to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    # 5) Optimizer / Scheduler / Loss

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

    # 6) Train (pass resume_from; trainer.load_checkpoint already uses map_location=device)
    trained_model = train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        resume_from=resume_from,
    )

    # 7) Final evaluation
    print("\n== Final evaluation on test split ==")
    _, loss = test_model(trained_model, val_loader, device, criterion)
    print(f"Test total loss: {loss:.4f}")


if __name__ == "__main__":
    main()
