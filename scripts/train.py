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

def main():
    # 1) Load config and set device
    cfg = load_config("configs/train_config.yaml")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Determine num_labels dynamically from the train DataFrame
    df_train = load_split(cfg, split="train")
    num_labels = len(df_train['label'].iloc[0])
    print(f"Detected num_labels = {num_labels}")

    # 3) Prepare DataLoaders
    train_loader = get_dataloader(cfg, split=["train", "train_nonelem"], sample_n=10000) 
    val_loader   = get_dataloader(cfg, split="test")

    # 4) Load vocab and instantiate model
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
    
    # 5) Set up optimizer, scheduler (optional), and loss
    print("weight decay:", cfg.training.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=float(cfg.training.weight_decay))
    total_steps = len(train_loader) * cfg.training.epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.training.learning_rate,
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy='cos'
    )
    
    criterion = nn.MSELoss(reduction='none')

    # 6) Run full training loop (with checkpointing & early stopping)
    trained_model = train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device
    )

    # 7) Final evaluation on the test split
    print("\n== Final evaluation on test split ==")
    _, loss = test_model(
        trained_model,
        val_loader,
        device,
        criterion
    )
    print(f"Test total loss: {loss:.4f}")

if __name__ == "__main__":
    main()