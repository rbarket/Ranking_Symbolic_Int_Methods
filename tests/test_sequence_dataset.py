import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from src.data.loader import split_train_validation
from src.data.dataset import (
    TokenSequenceDataset,
    token_sequence_collate_fn,
)
from src.models.registry import get_model_spec


def make_config(model_type="transformer"):
    return SimpleNamespace(
        model=SimpleNamespace(type=model_type),
        training=SimpleNamespace(
            batch_size=2,
            validation_fraction=0.25,
            split_seed=1998,
        ),
        data=SimpleNamespace(num_workers=0),
    )


def make_frame(size=8):
    return pd.DataFrame(
        {
            "prefix": [["[CLS]", f"token_{index}"] for index in range(size)],
            "label": [[float(index), -1.0] for index in range(size)],
            "source": ["elementary" if index % 2 == 0 else "nonelementary"
                       for index in range(size)],
        }
    )


def make_vocab(size=8):
    vocab = {"<pad>": 0, "<OOV>": 1, "[CLS]": 2}
    vocab.update({f"token_{index}": index + 3 for index in range(size)})
    return vocab


class TokenSequenceDatasetTests(unittest.TestCase):
    def test_collation_pads_tokens_and_returns_four_item_batch(self):
        batch = [
            (
                torch.tensor([2, 3, 4]),
                torch.tensor([1.0, -1.0]),
                torch.tensor([True, False]),
            ),
            (
                torch.tensor([2, 5]),
                torch.tensor([2.0, 3.0]),
                torch.tensor([True, True]),
            ),
        ]

        token_ids, token_mask, labels, label_mask = token_sequence_collate_fn(batch)

        self.assertEqual(tuple(token_ids.shape), (2, 3))
        self.assertTrue(torch.equal(token_ids[1], torch.tensor([2, 5, 0])))
        self.assertTrue(torch.equal(token_mask[1], torch.tensor([False, False, True])))
        self.assertEqual(tuple(labels.shape), (2, 2))
        self.assertEqual(tuple(label_mask.shape), (2, 2))

    def test_train_validation_split_is_deterministic_disjoint_and_complete(self):
        frame = make_frame()
        cfg = make_config()

        with patch("src.data.loader.load_split", return_value=frame):
            train_a, validation_a = split_train_validation(cfg)
            train_b, validation_b = split_train_validation(cfg)

        self.assertEqual(train_a.index.tolist(), train_b.index.tolist())
        self.assertEqual(validation_a.index.tolist(), validation_b.index.tolist())
        self.assertTrue(set(train_a.index).isdisjoint(validation_a.index))
        self.assertEqual(set(train_a.index) | set(validation_a.index), set(frame.index))
        self.assertEqual((len(train_a), len(validation_a)), (6, 2))

    def test_source_filtering_and_sampling_are_deterministic(self):
        frame = make_frame()
        with patch("src.data.dataset.load_vocab", return_value=make_vocab()):
            dataset = TokenSequenceDataset(
                make_config(),
                dataframe=frame,
                sample_n=2,
            )
            first_sample = [item[0].tolist() for item in dataset.data]
            dataset.set_data_type(None)
            second_sample = [item[0].tolist() for item in dataset.data]
            self.assertEqual(first_sample, second_sample)

            dataset.set_data_type("elementary")
            self.assertEqual(len(dataset), 2)
            elementary_ids = {make_vocab()[f"token_{index}"] for index in range(0, 8, 2)}
            self.assertTrue(all(item[0][1].item() in elementary_ids for item in dataset))

    def test_transformer_and_lstm_loaders_never_request_tree_positions(self):
        frame = make_frame(size=4)
        vocab = make_vocab(size=4)

        for model_type in ("transformer", "lstm"):
            with self.subTest(model_type=model_type), patch(
                "src.data.dataset.load_split",
                return_value=frame,
            ), patch(
                "src.data.dataset.load_vocab",
                return_value=vocab,
            ), patch(
                "src.data.dataset.load_precomputed_positions",
                side_effect=AssertionError("tree positions must not be loaded"),
            ):
                loader = get_model_spec(make_config(model_type)).build_dataloader(
                    make_config(model_type),
                    split="test",
                    sample_n=2,
                )
                batch = next(iter(loader))

                self.assertIsInstance(loader.dataset, TokenSequenceDataset)
                self.assertEqual(len(batch), 4)


if __name__ == "__main__":
    unittest.main()
