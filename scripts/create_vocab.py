from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


SPECIAL_TOKENS = ("<pad>", "<OOV>", "[CLS]")


def collect_prefix_tokens(dataset_path: Path) -> set[str]:
    """Return every token present in a parquet dataset's prefix column."""
    tokens = set()
    parquet_file = pq.ParquetFile(dataset_path)
    for batch in parquet_file.iter_batches(columns=["prefix"], batch_size=65_536):
        flattened = pc.list_flatten(batch.column(0))
        tokens.update(flattened.to_pylist())
    return tokens


def build_deterministic_vocab(tokens) -> dict[str, int]:
    """Assign stable IDs independent of dataset row order."""
    token_set = set(tokens)
    token_set.difference_update(SPECIAL_TOKENS)
    ordered_tokens = [*SPECIAL_TOKENS, *sorted(token_set)]
    return {token: token_id for token_id, token in enumerate(ordered_tokens)}


def write_vocab(path: Path, vocab: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(vocab, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def create_vocab(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    *,
    print_order: bool = True,
) -> dict[str, int]:
    train_tokens = collect_prefix_tokens(train_path)
    vocab = build_deterministic_vocab(train_tokens)
    write_vocab(output_path, vocab)

    test_tokens = collect_prefix_tokens(test_path)
    unseen_test_tokens = sorted(test_tokens - set(vocab))

    print(f"Wrote {output_path} with {len(vocab)} tokens from {train_path}")
    if unseen_test_tokens:
        print(
            "Test-only tokens mapped to <OOV>: "
            + ", ".join(unseen_test_tokens)
        )
    else:
        print(f"All tokens in {test_path} are covered by the training vocabulary.")

    if print_order:
        print("Vocabulary order:")
        for token, token_id in vocab.items():
            print(f"{token_id:>3}  {token}")

    return vocab


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a deterministic vocabulary from a configured training dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/train_tree_transformer_config.yaml",
        help="Config providing data.input_dir and data.vocab_path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path overriding data.vocab_path from the config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = Path(cfg.data.input_dir)
    output_path = Path(args.output or cfg.data.vocab_path)
    create_vocab(
        train_path=input_dir / "train_data.parquet",
        test_path=input_dir / "test_data.parquet",
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
