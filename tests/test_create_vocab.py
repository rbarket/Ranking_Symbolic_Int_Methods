import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.create_vocab import build_deterministic_vocab, create_vocab


class CreateVocabTests(unittest.TestCase):
    def test_order_is_stable_and_special_tokens_are_fixed(self):
        first = build_deterministic_vocab(["z", "a", "[CLS]", "m"])
        second = build_deterministic_vocab(["m", "z", "a", "[CLS]"])

        expected = {
            "<pad>": 0,
            "<OOV>": 1,
            "[CLS]": 2,
            "a": 3,
            "m": 4,
            "z": 5,
        }
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_create_vocab_uses_training_tokens_and_writes_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train_data.parquet"
            test_path = root / "test_data.parquet"
            output_path = root / "vocab.json"
            pd.DataFrame(
                {"prefix": [["[CLS]", "beta"], ["[CLS]", "alpha"]]}
            ).to_parquet(train_path)
            pd.DataFrame(
                {"prefix": [["[CLS]", "alpha"], ["[CLS]", "test_only"]]}
            ).to_parquet(test_path)

            vocab = create_vocab(
                train_path,
                test_path,
                output_path,
                print_order=False,
            )

            self.assertEqual(
                vocab,
                {
                    "<pad>": 0,
                    "<OOV>": 1,
                    "[CLS]": 2,
                    "alpha": 3,
                    "beta": 4,
                },
            )
            self.assertEqual(json.loads(output_path.read_text()), vocab)


if __name__ == "__main__":
    unittest.main()
