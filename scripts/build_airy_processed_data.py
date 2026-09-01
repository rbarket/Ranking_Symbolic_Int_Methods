import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.tree_utils import find_invalid_expressions
from src.data.preprocess import min_max_scale, replace_int_with_c
from scripts.create_vocab import create_vocab

RAW_AIRY_PATH = ROOT / "data" / "raw" / "nonelementary" / "AiryExamples_special_multiple_answers.json"
BASE_PROCESSED_DIR = ROOT / "data" / "processed"
OLD_PROCESSED_DIR = BASE_PROCESSED_DIR / "old"
AIRY_PROCESSED_DIR = BASE_PROCESSED_DIR / "airy"
VOCAB_PATH = ROOT / "data" / "vocab.json"

PROCESSED_COLUMNS = ["integrand", "prefix", "integral", "label_original", "source", "label"]
TEST_RAW_INDICES = [0, -1, -2]
EXPECTED_RAW_ROWS = 104
LOOKUP_LABEL_INDEX = 10


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def validate_raw_airy(raw_rows):
    if len(raw_rows) != EXPECTED_RAW_ROWS:
        raise ValueError(f"Expected {EXPECTED_RAW_ROWS} Airy rows, found {len(raw_rows)}")
    for idx, row in enumerate(raw_rows):
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError(f"Airy row {idx} must be a list of length 5")
        if not isinstance(row[0], str) or not row[0]:
            raise ValueError(f"Airy row {idx} integrand must be a non-empty string")
        if not isinstance(row[1], list):
            raise ValueError(f"Airy row {idx} input prefix must be a list")
        if not isinstance(row[2], str) or not row[2]:
            raise ValueError(f"Airy row {idx} integral must be a non-empty string")
        if not isinstance(row[3], list):
            raise ValueError(f"Airy row {idx} integral prefix must be a list")
        if not isinstance(row[4], list) or len(row[4]) != 13:
            raise ValueError(f"Airy row {idx} labels must be a list of length 13")
        if any(not isinstance(value, int) for value in row[4]):
            raise ValueError(f"Airy row {idx} labels must all be integers")


def process_airy_row(row):
    labels = list(np.delete(row[4], LOOKUP_LABEL_INDEX))
    prefix = ["[CLS]"] + replace_int_with_c(row[1])
    return {
        "integrand": row[0],
        "prefix": prefix,
        "integral": row[2],
        "label_original": labels,
        "source": "nonelementary",
        "label": min_max_scale(labels),
    }


def validate_processed_frame(frame, name, *, validate_prefixes):
    if list(frame.columns) != PROCESSED_COLUMNS:
        raise ValueError(f"{name} columns do not match expected schema: {list(frame.columns)}")
    label_original_lengths = frame["label_original"].apply(len)
    label_lengths = frame["label"].apply(len)
    if not label_original_lengths.eq(12).all():
        raise ValueError(f"{name} contains label_original rows that are not length 12")
    if not label_lengths.eq(12).all():
        raise ValueError(f"{name} contains label rows that are not length 12")
    if not frame["prefix"].apply(lambda prefix: len(prefix) > 0 and prefix[0] == "[CLS]").all():
        raise ValueError(f"{name} contains prefixes that do not start with [CLS]")
    if validate_prefixes:
        invalid = find_invalid_expressions(frame["prefix"])
        if invalid:
            raise ValueError(f"{name} contains {len(invalid)} invalid prefix expressions")


def resolve_test_indices(raw_rows):
    resolved = []
    for idx in TEST_RAW_INDICES:
        resolved_idx = idx if idx >= 0 else len(raw_rows) + idx
        if resolved_idx < 0 or resolved_idx >= len(raw_rows):
            raise ValueError(f"Requested test index {idx} resolves outside Airy data")
        resolved.append(resolved_idx)
    return resolved


def main():
    raw_rows = load_json(RAW_AIRY_PATH)
    validate_raw_airy(raw_rows)

    airy_rows = [process_airy_row(row) for row in raw_rows]
    airy_frame = pd.DataFrame(airy_rows, columns=PROCESSED_COLUMNS)
    validate_processed_frame(airy_frame, "Airy processed frame", validate_prefixes=True)

    base_train = pd.read_parquet(OLD_PROCESSED_DIR / "train_data.parquet")
    base_test = pd.read_parquet(OLD_PROCESSED_DIR / "test_data.parquet")
    validate_processed_frame(base_train, "Base train frame", validate_prefixes=False)
    validate_processed_frame(base_test, "Base test frame", validate_prefixes=False)

    resolved_test_indices = resolve_test_indices(raw_rows)
    airy_test_frame = airy_frame.iloc[resolved_test_indices].reset_index(drop=True)

    train_out = pd.concat([base_train, airy_frame], ignore_index=True)
    test_out = pd.concat([base_test, airy_test_frame], ignore_index=True)
    validate_processed_frame(train_out, "Airy train output", validate_prefixes=False)
    validate_processed_frame(test_out, "Airy test output", validate_prefixes=False)

    expected_train_rows = len(base_train) + len(airy_frame)
    expected_test_rows = len(base_test) + len(airy_test_frame)
    if len(train_out) != expected_train_rows:
        raise ValueError(f"Expected {expected_train_rows} train rows, found {len(train_out)}")
    if len(test_out) != expected_test_rows:
        raise ValueError(f"Expected {expected_test_rows} test rows, found {len(test_out)}")

    AIRY_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = AIRY_PROCESSED_DIR / "train_data.parquet"
    test_path = AIRY_PROCESSED_DIR / "test_data.parquet"
    train_out.to_parquet(train_path)
    test_out.to_parquet(test_path)
    vocab = create_vocab(train_path, test_path, VOCAB_PATH)

    manifest = {
        "raw_airy_path": str(RAW_AIRY_PATH.relative_to(ROOT)),
        "base_train_rows": len(base_train),
        "base_test_rows": len(base_test),
        "airy_train_rows_added": len(airy_frame),
        "airy_test_raw_indices_requested": TEST_RAW_INDICES,
        "airy_test_raw_indices_resolved": resolved_test_indices,
        "train_rows": len(train_out),
        "test_rows": len(test_out),
        "processed_columns": PROCESSED_COLUMNS,
        "lookup_label_index_removed": LOOKUP_LABEL_INDEX,
        "vocab_path": str(VOCAB_PATH.relative_to(ROOT)),
        "vocab_size": len(vocab),
        "vocab_order": "special tokens followed by lexical token order",
    }
    write_json(AIRY_PROCESSED_DIR / "manifest.json", manifest)

    print(f"Wrote {train_path} ({len(train_out)} rows)")
    print(f"Wrote {test_path} ({len(test_out)} rows)")
    print(f"Wrote {VOCAB_PATH}")
    print(f"Wrote {AIRY_PROCESSED_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
