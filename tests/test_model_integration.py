import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn

from src.data.dataset import PrefixExpressionDataset
from src.data.tree_lstm_dataset import TreeLSTMGraphDataset, tree_lstm_collate_fn
from src.models.lstm import LSTM
from src.models.registry import get_model_spec
from src.models.transformer import Transformer
from src.models.tree_lstm import TreeLSTM
from src.models.tree_transformer import TreeTransformer
from src.training.evaluation import test_model
from src.training.trainer import train, train_epoch


def make_config(model_type="transformer"):
    model = SimpleNamespace(
        d_model=8,
        layers=1,
        heads=2,
        dim_feedforward=16,
        dropout=0.0,
        activation="gelu",
        max_seq_len=16,
        hidden_size=8,
    )
    if model_type is not None:
        model.type = model_type

    return SimpleNamespace(
        model=model,
        tree=SimpleNamespace(branching_factor=2, depth=4),
        training=SimpleNamespace(
            batch_size=2,
            validation_fraction=0.1,
            split_seed=1998,
        ),
        data=SimpleNamespace(num_workers=0),
    )


def make_tree_batch(batch_size=1):
    token_ids = torch.tensor([[2, 3, 4]]).repeat(batch_size, 1)
    positions = torch.zeros(batch_size, 3, 8)
    token_mask = torch.zeros(batch_size, 3, dtype=torch.bool)
    labels = torch.tensor([[1.0, 2.0, 3.0]]).repeat(batch_size, 1)
    label_mask = torch.tensor([[True, False, True]]).repeat(batch_size, 1)
    return token_ids, positions, token_mask, labels, label_mask


def make_token_batch(batch_size=1):
    token_ids, _, token_mask, labels, label_mask = make_tree_batch(batch_size)
    return token_ids, token_mask, labels, label_mask


def make_batch(model_type, batch_size=1):
    if model_type == "tree_transformer":
        return make_tree_batch(batch_size)
    return make_token_batch(batch_size)


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class ModelRegistryTests(unittest.TestCase):
    def test_selects_all_integrated_models_and_defaults_to_tree_transformer(self):
        expected_classes = {
            "tree_transformer": TreeTransformer,
            "transformer": Transformer,
            "lstm": LSTM,
        }
        for model_type, expected_class in expected_classes.items():
            with self.subTest(model_type=model_type):
                cfg = make_config(model_type)
                spec = get_model_spec(cfg)
                model = spec.build_model(cfg, vocab_size=20, num_labels=3)
                self.assertEqual(spec.model_type, model_type)
                self.assertIsInstance(model, expected_class)

        default_cfg = make_config(None)
        default_spec = get_model_spec(default_cfg)
        default_model = default_spec.build_model(
            default_cfg,
            vocab_size=20,
            num_labels=3,
        )
        self.assertEqual(default_spec.model_type, "tree_transformer")
        self.assertIsInstance(default_model, TreeTransformer)

        tree_lstm_cfg = make_config("tree_lstm")
        with patch("src.models.registry.import_module"):
            tree_lstm_spec = get_model_spec(tree_lstm_cfg)
        tree_lstm = tree_lstm_spec.build_model(
            tree_lstm_cfg,
            vocab_size=20,
            num_labels=3,
        )
        self.assertEqual(tree_lstm_spec.model_type, "tree_lstm")
        self.assertIsInstance(tree_lstm, TreeLSTM)

    def test_resolves_architecture_specific_hyperparameters(self):
        transformer_spec = get_model_spec(make_config("transformer"))
        transformer_params = transformer_spec.resolve_hyperparameters(
            make_config("transformer"),
            vocab_size=20,
            num_labels=3,
        )
        self.assertEqual(transformer_params["max_seq_len"], 16)
        self.assertNotIn("n", transformer_params)

        tree_spec = get_model_spec(make_config("tree_transformer"))
        tree_params = tree_spec.resolve_hyperparameters(
            make_config("tree_transformer"),
            vocab_size=20,
            num_labels=3,
        )
        self.assertEqual(tree_params["n"], 2)
        self.assertEqual(tree_params["k"], 4)
        self.assertNotIn("max_seq_len", tree_params)

        lstm_spec = get_model_spec(make_config("lstm"))
        lstm_params = lstm_spec.resolve_hyperparameters(
            make_config("lstm"),
            vocab_size=20,
            num_labels=3,
        )
        self.assertEqual(lstm_params["d_model"], 8)
        self.assertEqual(lstm_params["num_layers"], 1)
        self.assertNotIn("nhead", lstm_params)
        self.assertNotIn("max_seq_len", lstm_params)

        with patch("src.models.registry.import_module"):
            tree_lstm_spec = get_model_spec(make_config("tree_lstm"))
        tree_lstm_params = tree_lstm_spec.resolve_hyperparameters(
            make_config("tree_lstm"),
            vocab_size=20,
            num_labels=3,
        )
        self.assertEqual(tree_lstm_params["hidden_size"], 8)
        self.assertNotIn("num_layers", tree_lstm_params)

    def test_tree_lstm_dependency_and_data_parallel_errors_are_clear(self):
        with patch(
            "src.models.registry.import_module",
            side_effect=ImportError("missing dgl"),
        ), self.assertRaisesRegex(RuntimeError, "requires DGL"):
            get_model_spec(make_config("tree_lstm"))

        with patch("src.models.registry.import_module"):
            spec = get_model_spec(make_config("tree_lstm"))
        with self.assertRaisesRegex(ValueError, "without --data_parallel"):
            spec.validate_data_parallel(True)

    def test_rejects_unknown_model_types(self):
        with self.assertRaisesRegex(ValueError, "Unknown model.type"):
            get_model_spec(make_config("convolution"))

    def test_architecture_adapters_run_and_mask_all_integrated_models(self):
        for model_type in ("tree_transformer", "transformer", "lstm"):
            with self.subTest(model_type=model_type):
                cfg = make_config(model_type)
                spec = get_model_spec(cfg)
                model = spec.build_model(cfg, vocab_size=20, num_labels=3)
                result = spec.forward_batch(
                    model,
                    make_batch(model_type),
                    torch.device("cpu"),
                    apply_label_mask=True,
                )

                self.assertEqual(result.predictions.shape, (1, 3))
                self.assertEqual(result.labels.shape, (1, 3))
                self.assertEqual(result.predictions[0, 1].item(), 0.0)

    def test_batch_size_one_trains_evaluates_and_steps_per_batch(self):
        for model_type in ("tree_transformer", "transformer", "lstm"):
            with self.subTest(model_type=model_type):
                cfg = make_config(model_type)
                spec = get_model_spec(cfg)
                model = spec.build_model(cfg, vocab_size=20, num_labels=3)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                scheduler = CountingScheduler()
                loader = [make_batch(model_type), make_batch(model_type)]

                train_loss = train_epoch(
                    model,
                    loader,
                    optimizer,
                    torch.device("cpu"),
                    log_interval=10,
                    batch_forward=spec.forward_batch,
                    scheduler=scheduler,
                )
                outputs, eval_loss = test_model(
                    model,
                    [make_batch(model_type)],
                    torch.device("cpu"),
                    nn.MSELoss(reduction="none"),
                    spec.forward_batch,
                )

                self.assertTrue(torch.isfinite(torch.tensor(train_loss)))
                self.assertTrue(torch.isfinite(torch.tensor(eval_loss)))
                self.assertEqual(outputs[0].shape, (1, 3))
                self.assertEqual(scheduler.steps, len(loader))

    def test_tree_lstm_adapter_trains_and_evaluates_with_batch_size_one(self):
        try:
            import dgl  # noqa: F401
        except ImportError:
            self.skipTest("DGL is not installed in this environment")

        cfg = make_config("tree_lstm")
        frame = pd.DataFrame(
            {
                "prefix": [["[CLS]", "sub", "x", "1"]],
                "label": [[0.1, -1.0, 0.9]],
                "source": ["elementary"],
            }
        )
        vocab = {
            "<pad>": 0,
            "<OOV>": 1,
            "[CLS]": 2,
            "sub": 3,
            "x": 4,
            "1": 5,
        }
        with patch(
            "src.data.tree_lstm_dataset.load_vocab",
            return_value=vocab,
        ):
            dataset = TreeLSTMGraphDataset(cfg, dataframe=frame)
            batch = tree_lstm_collate_fn([dataset[0]])

        spec = get_model_spec(cfg)
        model = spec.build_model(cfg, vocab_size=len(vocab), num_labels=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_loss = train_epoch(
            model,
            [batch, batch],
            optimizer,
            torch.device("cpu"),
            log_interval=10,
            batch_forward=spec.forward_batch,
        )
        outputs, eval_loss = test_model(
            model,
            [batch],
            torch.device("cpu"),
            nn.MSELoss(reduction="none"),
            spec.forward_batch,
        )

        self.assertTrue(torch.isfinite(torch.tensor(train_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(eval_loss)))
        self.assertEqual(outputs[0].shape, (1, 3))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_training_writes_epoch_metrics_and_restores_best_checkpoint(self):
        cfg = make_config("transformer")
        cfg.experiment_name = "metrics_test"
        cfg.training.epochs = 1
        cfg.training.log_interval = 10
        cfg.training.early_stop_patience = 2
        spec = get_model_spec(cfg)
        params = spec.resolve_hyperparameters(cfg, vocab_size=20, num_labels=3)
        model = spec.model_class(**params)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg.paths = SimpleNamespace(save_dir=temp_dir)
            trained = train(
                cfg=cfg,
                model=model,
                train_loader=[make_token_batch()],
                val_loader=[make_token_batch()],
                optimizer=optimizer,
                scheduler=None,
                criterion=nn.MSELoss(reduction="none"),
                device=torch.device("cpu"),
                model_spec=spec,
                model_hyperparameters=params,
            )

            metrics_path = os.path.join(temp_dir, "metrics_test_metrics.csv")
            best_path = os.path.join(temp_dir, "metrics_test_best.pth")
            with open(metrics_path, newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], "1")
            self.assertIn("validation_smallest_label_accuracy", rows[0])
            best_state = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )["model_state_dict"]
            for name, value in trained.state_dict().items():
                torch.testing.assert_close(value, best_state[name])

    def test_sample_n_limits_non_training_splits(self):
        frame = pd.DataFrame(
            {
                "source": ["elementary"] * 4 + ["nonelementary"] * 2,
            }
        )

        def fake_build_data(dataset, selected_frame):
            dataset.data = list(selected_frame.index)

        with patch("src.data.dataset.load_split", return_value=frame), patch.object(
            PrefixExpressionDataset,
            "_build_data",
            new=fake_build_data,
        ):
            dataset = PrefixExpressionDataset(
                make_config("transformer"),
                split="test",
                sample_n=2,
            )
            self.assertEqual(len(dataset), 2)

            dataset.set_data_type("elementary")
            self.assertEqual(len(dataset), 2)


if __name__ == "__main__":
    unittest.main()
