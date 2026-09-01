import math
import unittest

import torch

from src.models.transformer import Transformer


class TransformerTests(unittest.TestCase):
    def make_model(self, **overrides):
        kwargs = {
            "vocab_size": 20,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 16,
            "num_labels": 3,
            "max_seq_len": 16,
            "dropout": 0.0,
        }
        kwargs.update(overrides)
        return Transformer(**kwargs)

    def test_trainer_style_call_shape_masks_labels_and_backpropagates(self):
        model = self.make_model()
        token_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
        tree_positions = torch.randn(2, 4, 8)
        token_mask = token_ids.eq(0)
        label_mask = torch.tensor([[True, False, True], [False, True, True]])
        attn_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

        output = model(
            token_ids,
            tree_positions,
            token_mask,
            label_mask,
            attn_mask,
        )

        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.equal(output[~label_mask], torch.zeros(2)))
        output.sum().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_sinusoidal_encoding_matches_known_values(self):
        model = self.make_model(d_model=4, nhead=2, num_layers=1)

        expected_zero = torch.tensor([0.0, 1.0, 0.0, 1.0])
        expected_one = torch.tensor(
            [math.sin(1.0), math.cos(1.0), math.sin(0.01), math.cos(0.01)]
        )

        torch.testing.assert_close(model.pos_encoding[0], expected_zero)
        torch.testing.assert_close(model.pos_encoding[1], expected_one)

    def test_external_tree_positions_are_ignored(self):
        model = self.make_model().eval()
        token_ids = torch.tensor([[2, 3, 4]])
        token_mask = torch.zeros_like(token_ids, dtype=torch.bool)

        with torch.no_grad():
            zero_positions = model(
                token_ids,
                torch.zeros(1, 3, 8),
                token_mask,
            )
            random_positions = model(
                token_ids,
                torch.randn(1, 3, 8) * 1000,
                token_mask,
            )

        torch.testing.assert_close(zero_positions, random_positions)

    def test_padding_mask_preserves_root_output(self):
        model = self.make_model().eval()
        short_tokens = torch.tensor([[2, 3, 4]])
        padded_tokens = torch.tensor([[2, 3, 4, 7, 8]])

        with torch.no_grad():
            short_output = model(
                short_tokens,
                None,
                torch.zeros_like(short_tokens, dtype=torch.bool),
            )
            padded_output = model(
                padded_tokens,
                None,
                torch.tensor([[False, False, False, True, True]]),
            )

        torch.testing.assert_close(short_output, padded_output, rtol=1e-5, atol=1e-6)

    def test_rejects_sequences_longer_than_configured_table(self):
        model = self.make_model(max_seq_len=3)
        token_ids = torch.tensor([[2, 3, 4, 5]])

        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len=3"):
            model(token_ids, None, torch.zeros_like(token_ids, dtype=torch.bool))

if __name__ == "__main__":
    unittest.main()
