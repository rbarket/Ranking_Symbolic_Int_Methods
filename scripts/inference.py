import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from src.utils.config import load_config
from src.utils.io import load_vocab, load_precomputed_positions
from src.data.dataset import PrefixExpressionDataset, collate_fn
from src.models.tree_transformer import TreeTransformer
from src.training.evaluation import test_model

def main(checkpoint_path: str, split: str = "test", sample_n: int = None):
    # 1) Load config & device
    cfg = load_config("configs/train_config.yaml")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Build dataset & loader
    ds_elem = PrefixExpressionDataset(cfg, split='test', sample_n=sample_n)
    ds_nonelem = PrefixExpressionDataset(cfg, split='test_nonelem', sample_n=sample_n)
    ds_combined = ConcatDataset([ds_elem, ds_nonelem])
    
    datasets = {
        "elementary": ds_elem,
        "nonelementary": ds_nonelem,
        "combined": ds_combined
    }
    
    print(type(ds_elem), type(ds_nonelem), type(ds_combined))
    
    loader = DataLoader(
        ds_combined,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    # 3) Recreate your model architecture
    vocab = load_vocab(cfg)
    
    # infer num_labels from the first item
    dummy_labels = ds_elem.data[0][2]
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
    ).to(device)

    # 4) Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # if you saved a full state dict:
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    for ds_name, ds in datasets.items():
        print(f"Evaluating on dataset: {ds_name} with {len(ds)} examples")
    
        loader = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
            )
    
        # test the model
        all_preds, _ = test_model( # don't need the loss 
            model=model,
            test_loader=loader,
            device=device,
            criterion=nn.MSELoss(reduction='none')  # Use the same loss as during training
            ) 

    # Save results for only combined dataset (last item in for loop)
    all_preds = torch.cat(all_preds, dim=0)  # [N, num_labels]
    print(f"Inference complete: {all_preds.shape}")

    # (Optional) Save to disk
    out_path = cfg.paths.save_dir + "/inference_outputs.pt"
    torch.save(all_preds, out_path)
    print(f"Saved all predictions to {out_path}")

if __name__ == "__main__":
    # Example usage:
    # python -m scripts.inference \
    #    --checkpoint_path experiments/baseline_tree_model/baseline_tree_model_best.pth
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--sample_n", type=int, default=None,
                   help="If set, runs inference on a random subset of this many examples")
    args = p.parse_args()
    main(args.checkpoint_path, split=args.split, sample_n=args.sample_n)
