import unittest

import torch

from src.models.lstm import LSTM


class LSTMTests(unittest.TestCase):
    def make_model(self, **overrides):
        kwargs = {
            "vocab_size": 20,
            "d_model": 8,
            "num_layers": 2,
            "num_labels": 3,
            "dropout": 0.0,
        }
        kwargs.update(overrides)
        return LSTM(**kwargs)

    def test_output_shape_label_mask_and_backpropagation(self):
        model = self.make_model()
        token_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
        token_mask = token_ids.eq(0)
        label_mask = torch.tensor([[True, False, True], [False, True, True]])

        output = model(token_ids, token_mask, label_mask)

        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.equal(output[~label_mask], torch.zeros(2)))
        output.sum().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_right_padding_does_not_change_prediction(self):
        model = self.make_model().eval()
        short_tokens = torch.tensor([[2, 3, 4]])
        padded_tokens = torch.tensor([[2, 3, 4, 7, 8]])

        with torch.no_grad():
            short_output = model(
                short_tokens,
                torch.zeros_like(short_tokens, dtype=torch.bool),
            )
            padded_output = model(
                padded_tokens,
                torch.tensor([[False, False, False, True, True]]),
            )

        torch.testing.assert_close(short_output, padded_output)

    def test_unsorted_sequence_lengths_match_individual_predictions(self):
        model = self.make_model().eval()
        token_ids = torch.tensor(
            [
                [2, 3, 0, 0],
                [4, 5, 6, 7],
                [8, 9, 10, 0],
            ]
        )
        token_mask = token_ids.eq(0)

        with torch.no_grad():
            batched_output = model(token_ids, token_mask)
            individual_outputs = []
            for tokens, length in zip(token_ids, (~token_mask).sum(dim=1)):
                tokens = tokens[:length].unsqueeze(0)
                mask = torch.zeros_like(tokens, dtype=torch.bool)
                individual_outputs.append(model(tokens, mask))

        torch.testing.assert_close(batched_output, torch.cat(individual_outputs, dim=0))

    def test_internal_dropout_depends_on_layer_count(self):
        single_layer = self.make_model(num_layers=1, dropout=0.25)
        multiple_layers = self.make_model(num_layers=3, dropout=0.25)

        self.assertEqual(single_layer.lstm.dropout, 0.0)
        self.assertEqual(multiple_layers.lstm.dropout, 0.25)
        self.assertEqual(single_layer.cls_dropout.p, 0.25)

    def test_rejects_entirely_padded_sequence(self):
        model = self.make_model()
        token_ids = torch.tensor([[0, 0], [2, 0]])
        token_mask = token_ids.eq(0)

        with self.assertRaisesRegex(ValueError, "at least one non-padding token"):
            model(token_ids, token_mask)

    def test_rejects_mismatched_token_mask_shape(self):
        model = self.make_model()
        token_ids = torch.tensor([[2, 3, 4]])
        token_mask = torch.tensor([[False, False]])

        with self.assertRaisesRegex(ValueError, "same shape"):
            model(token_ids, token_mask)


if __name__ == "__main__":
    unittest.main()
