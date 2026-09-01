import os
import argparse
import json
import hashlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import config_from_dict, load_config, resolve_device
from src.utils.io import load_vocab, load_vocab_path
from src.utils.checkpoint import (
    checkpoint_state_dict,
    is_v2_checkpoint,
    load_checkpoint_payload,
    resize_embedding_for_vocab,
    validate_checkpoint_model_type,
    validate_embedded_vocabulary,
    validate_vocab_checkpoint_compatible,
    vocabulary_sha256,
)
from src.data.dataset import load_split
from src.models.registry import get_model_spec
from src.training.trainer import train
from src.training.evaluation import test_model


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="Path to config file. Optional for version-2 resume checkpoints.")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint to resume from (overrides config).")
    p.add_argument("--init_from", type=str, default=None,
                   help="Path to checkpoint weights to initialize from without resuming optimizer/epoch.")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | cuda:{idx}")
    p.add_argument("--data_parallel", action="store_true",
                   help="Use DataParallel when multiple GPUs are available (off by default).")
    p.add_argument("--n", type=int, default=None,
                   help="Number of training samples to use (overrides config.training.n if set).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of epochs to run (overrides config.training.epochs if set).")
    p.add_argument("--eval_n", type=int, default=None,
                   help="If set, limit both validation and final-test samples.")
    p.add_argument("--input_dir", default=None,
                   help="Runtime override for data.input_dir.")
    p.add_argument("--vocab_path", default=None,
                   help="Runtime vocabulary override (must match for resume).")
    p.add_argument("--save_dir", default=None,
                   help="Runtime override for checkpoint and metrics output.")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def dataframe_sha256(dataframe) -> str:
    """Fingerprint the exact ordered rows used by a training/evaluation loader."""
    digest = hashlib.sha256()
    columns = [
        column for column in ("row_id", "prefix", "label", "source")
        if column in dataframe.columns
    ]
    for values in dataframe[columns].itertuples(index=False, name=None):
        digest.update(
            json.dumps(values, default=str, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def initialize_model_weights(
    model,
    init_from,
    vocab,
    cfg,
    device,
    expected_model_type,
):
    checkpoint = load_checkpoint_payload(init_from, map_location=device)
    validate_checkpoint_model_type(checkpoint, expected_model_type)
    state_dict = checkpoint_state_dict(checkpoint)

    allow_resize = bool(getattr(cfg.training, "resize_token_embeddings", False))
    embedded_old_vocab = None
    vocabulary_differs = False
    if is_v2_checkpoint(checkpoint):
        embedded_old_vocab = validate_embedded_vocabulary(checkpoint)
        vocabulary_differs = (
            vocabulary_sha256(embedded_old_vocab) != vocabulary_sha256(vocab)
        )
        if vocabulary_differs and not allow_resize:
            raise ValueError(
                "Initialization vocabulary differs from the checkpoint vocabulary. "
                "Enable the explicit embedding-resize path for intentional changes."
            )
    validate_vocab_checkpoint_compatible(state_dict, len(vocab), allow_resize=allow_resize)

    if state_dict["embedding.weight"].shape[0] != len(vocab) or vocabulary_differs:
        old_vocab_path = getattr(cfg.training, "init_vocab_path", None)
        if embedded_old_vocab is not None:
            old_vocab = embedded_old_vocab
            old_vocab_description = "the checkpoint's embedded vocabulary"
        elif old_vocab_path:
            old_vocab = load_vocab_path(old_vocab_path)
            old_vocab_description = old_vocab_path
        else:
            raise ValueError("training.init_vocab_path is required when resizing token embeddings.")
        state_dict = resize_embedding_for_vocab(state_dict, old_vocab, vocab)
        print(
            f"[init] resized embedding from {len(old_vocab)} to {len(vocab)} rows "
            f"using {old_vocab_description}"
        )

    (model.module if hasattr(model, "module") else model).load_state_dict(state_dict)
    print(f"[init] loaded model weights from {init_from}")


def main():
    # --- args & config ---
    args = parse_args()
    if args.resume_from and not os.path.isfile(args.resume_from):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.resume_from}")
    if args.init_from and not os.path.isfile(args.init_from):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.init_from}")

    resume_checkpoint = (
        load_checkpoint_payload(args.resume_from, map_location="cpu")
        if args.resume_from else None
    )
    if args.config is not None:
        cfg = load_config(args.config)
    elif resume_checkpoint is not None and is_v2_checkpoint(resume_checkpoint):
        cfg = config_from_dict(resume_checkpoint["config"])
        cfg.data.embedded_vocab = resume_checkpoint["vocabulary"]
        print("[resume] reconstructed configuration and vocabulary from checkpoint")
    else:
        cfg = load_config("configs/train_tree_transformer_config.yaml")

    if args.input_dir is not None:
        cfg.data.input_dir = args.input_dir
    if args.vocab_path is not None:
        cfg.data.vocab_path = args.vocab_path
        if hasattr(cfg.data, "embedded_vocab"):
            delattr(cfg.data, "embedded_vocab")
    if args.save_dir is not None:
        cfg.paths.save_dir = args.save_dir
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch_size must be at least 1.")
        cfg.training.batch_size = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num_workers cannot be negative.")
        cfg.data.num_workers = args.num_workers

    model_spec = get_model_spec(cfg)
    model_spec.validate_data_parallel(args.data_parallel)
    print(f"Selected model type: {model_spec.model_type}")

    training_seed = int(getattr(cfg.training, "seed", 1998))
    set_random_seed(training_seed)
    print(f"Random seed: {training_seed}")

    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1.")
        cfg.training.epochs = args.epochs
    if args.n is not None:
        if args.n < 1:
            raise ValueError("--n must be at least 1.")
        cfg.training.n = args.n
    if args.eval_n is not None and args.eval_n < 1:
        raise ValueError("--eval_n must be at least 1.")
    if args.eval_n is not None:
        cfg.training.eval_n = args.eval_n

    # Resolve resume path: CLI > config > None. A config-selected legacy resume is
    # loaded here after the configuration becomes available.
    resume_from = args.resume_from or getattr(getattr(cfg, "training", {}), "resume_from", None)
    init_from = args.init_from or getattr(getattr(cfg, "training", {}), "init_from", None)
    if resume_from and init_from:
        raise ValueError("Use either resume_from or init_from, not both.")
    if resume_from and not os.path.isfile(resume_from):
        raise FileNotFoundError(f"Checkpoint does not exist: {resume_from}")
    if init_from and not os.path.isfile(init_from):
        raise FileNotFoundError(f"Checkpoint does not exist: {init_from}")
    if resume_from and resume_checkpoint is None:
        resume_checkpoint = load_checkpoint_payload(resume_from, map_location="cpu")

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
    # The registry selects the architecture-appropriate dataloader builder.
    sample_n = args.n if args.n is not None else getattr(cfg.training, "n", None)
    eval_n = args.eval_n if args.eval_n is not None else getattr(
        cfg.training,
        "eval_n",
        None,
    )
    train_loader, val_loader = model_spec.build_train_validation_dataloaders(
        cfg,
        train_n=sample_n,
        eval_n=eval_n,
    )
    test_loader = model_spec.build_dataloader(
        cfg,
        split="test",
        sample_n=eval_n,
    )

    # 4) Model
    vocab = load_vocab(cfg)
    model_hyperparameters = model_spec.resolve_hyperparameters(
        cfg,
        vocab_size=len(vocab),
        num_labels=num_labels,
    )
    model = model_spec.build_model(cfg, len(vocab), num_labels)

    # Optional multi-GPU for training
    if dp_ok:
        model = nn.DataParallel(model)

    model = model.to(device)
    if init_from is not None:
        initialize_model_weights(
            model,
            init_from,
            vocab,
            cfg,
            device,
            expected_model_type=model_spec.model_type,
        )
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
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

    data_metadata = {
        "input_dir": str(cfg.data.input_dir),
        "vocab_path": str(getattr(cfg.data, "vocab_path", "")),
        "positions_path": str(getattr(cfg.data, "positions_path", "")),
        "batch_size": int(cfg.training.batch_size),
        "training_n": sample_n,
        "evaluation_n": eval_n,
        "validation_fraction": float(getattr(cfg.training, "validation_fraction", 0.1)),
        "split_seed": int(getattr(cfg.training, "split_seed", 1998)),
        "steps_per_epoch": len(train_loader),
        "train_rows": len(train_loader.dataset.selected_frame),
        "validation_rows": len(val_loader.dataset.selected_frame),
        "train_data_sha256": dataframe_sha256(train_loader.dataset.selected_frame),
        "validation_data_sha256": dataframe_sha256(val_loader.dataset.selected_frame),
    }

    # 6) Train (pass resume_from; trainer.load_checkpoint already uses map_location=device)
    training_started = time.perf_counter()
    trained_model = train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        model_spec=model_spec,
        model_hyperparameters=model_hyperparameters,
        resume_from=resume_from,
        vocabulary=vocab,
        data_metadata=data_metadata,
        resume_checkpoint=resume_checkpoint,
    )

    # 7) Final evaluation
    print("\n== Final evaluation on held-out test split ==")
    training_seconds = time.perf_counter() - training_started
    _, loss, test_metrics = test_model(
        trained_model,
        test_loader,
        device,
        criterion,
        model_spec.forward_batch,
        return_metrics=True,
    )
    print(f"Test total loss: {loss:.4f}")

    best_checkpoint_path = os.path.join(
        cfg.paths.save_dir,
        f"{getattr(cfg, 'experiment_name', 'model')}_best.pth",
    )
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = load_checkpoint_payload(best_checkpoint_path, map_location="cpu")
        best_validation_rank_loss = best_checkpoint["best_val_loss"]
    else:
        best_validation_rank_loss = None

    summary = {
        "experiment_name": getattr(cfg, "experiment_name", "model"),
        "model_type": model_spec.model_type,
        "model_hyperparameters": model_hyperparameters,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in trained_model.parameters()
            if parameter.requires_grad
        ),
        "training": {
            "learning_rate": float(cfg.training.learning_rate),
            "batch_size": int(cfg.training.batch_size),
            "epochs": int(cfg.training.epochs),
            "weight_decay": float(cfg.training.weight_decay),
            "training_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "seed": training_seed,
            "split_seed": int(getattr(cfg.training, "split_seed", 1998)),
        },
        "training_seconds": training_seconds,
        "best_validation_rank_loss": best_validation_rank_loss,
        "test_metrics": {
            "rank_loss": test_metrics.rank_loss,
            "masked_mse": test_metrics.masked_mse,
            "smallest_label_accuracy": (
                test_metrics.smallest_label_accuracy
            ),
            "correct_smallest": test_metrics.correct_smallest,
            "evaluated_samples": test_metrics.evaluated_samples,
        },
    }
    summary_path = os.path.join(
        cfg.paths.save_dir,
        f"{getattr(cfg, 'experiment_name', 'model')}_summary.json",
    )
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
