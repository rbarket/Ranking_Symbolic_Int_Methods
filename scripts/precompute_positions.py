# run this from root with "python -m scripts.precompute_positions"
import torch
from src.utils.config import load_config
from src.utils.tree_utils import push_down, precompute_all_positions, save_precomputed_positions

def main():
    # Load config
    cfg = load_config("configs/train_config.yaml")
    n = cfg.tree.branching_factor
    k = cfg.tree.depth

    print(f"Branching factor: {n}, Depth: {k}")
    # Compute positions
    precomputed_positions = precompute_all_positions(n, k)
    print("saving...")
    # Save to disk
    save_precomputed_positions(precomputed_positions, cfg)

if __name__ == "__main__":
    
    main()