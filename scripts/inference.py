import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.io import load_vocab
from src.data.dataset import PrefixExpressionDataset, collate_fn
from src.models.tree_transformer import TreeTransformer
from src.training.evaluation import test_model


def resolve_device(device_arg: str) -> torch.device:
    """
    device_arg: 'auto' | 'cpu' | 'cuda' | 'cuda:0' etc.
    """
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        # Respect explicit CUDA request but fall back to CPU if unavailable
        return torch.device(device_arg if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_strip_module_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix if present (from DataParallel checkpoints). Used if loading on CPU or single GPU."""
    if not state_dict:
        return state_dict
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def main(checkpoint_path: str, split: str = "test", sample_n: int = None,
         device_arg: str = "auto", data_parallel: bool = False):
    # 1) Load config & device
    cfg = load_config("configs/train_config.yaml")
    device = resolve_device(device_arg)
    print(f"Using device: {device}")

    # 2) Build dataset & loader
    ds = PrefixExpressionDataset(cfg, split=split, sample_n=sample_n)
    vocab = load_vocab(cfg)

    # infer num_labels from the first item
    dummy_labels = ds.data[0][2]
    num_labels = dummy_labels.shape[-1]

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

    # Load weights (works for CPU or CUDA)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # If the checkpoint was saved with DataParallel, keys may have 'module.' prefix
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = maybe_strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    # Optionally wrap with DataParallel for multi-GPU inference
    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )

    # Inference
    last_all_preds = None
    for source in ["elementary", "nonelementary", None]:  # None -> combined
        ds.set_data_type(source)
        print(f"Running inference on {source or 'combined'} data...")
        with torch.no_grad():
            all_preds, _ = test_model(
                model=model,
                test_loader=loader,
                device=device,
                criterion=nn.MSELoss(reduction='none')
            )
        last_all_preds = all_preds  # keep the combined (last loop) results

    # Save results for the combined dataset (last iteration)
    all_preds = torch.cat(last_all_preds, dim=0)  # [N, num_labels]
    print(f"Inference complete: {all_preds.shape}")

    out_path = cfg.paths.save_dir + "/inference_outputs.pt"
    torch.save(all_preds, out_path)
    print(f"Saved all predictions to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--sample_n", type=int, default=None,
                   help="If set, runs inference on a random subset of this many examples")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:{idx}")
    p.add_argument("--data_parallel", action="store_true",
                   help="Use DataParallel when multiple GPUs are available (off by default).")
    args = p.parse_args()

    main(
        checkpoint_path=args.checkpoint_path,
        split=args.split,
        sample_n=args.sample_n,
        device_arg=args.device,
        data_parallel=args.data_parallel
    )
