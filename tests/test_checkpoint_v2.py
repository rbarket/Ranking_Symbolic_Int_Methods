import json
import csv
import os
import random
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.inference import main as run_inference
from src.models.registry import get_model_spec_by_type
from src.training.trainer import _METRIC_FIELDS, _prepare_metrics_file, _resume_training_state
from src.training.evaluation import compute_evaluation_metrics
from src.utils.checkpoint import (
    build_v2_checkpoint,
    checkpoint_state_dict,
    load_checkpoint,
    load_checkpoint_payload,
    reconstruct_model_from_checkpoint,
    restore_rng_state,
    save_checkpoint_payload,
    save_checkpoint,
    validate_embedded_vocabulary,
    vocabulary_sha256,
)
from src.utils.config import DotDict, config_to_dict
from src.utils.tree_utils import precompute_all_positions


VOCAB = {"<pad>": 0, "<OOV>": 1, "[CLS]": 2, "x": 3, "sin": 4}


def hyperparameters(model_type):
    common = {"vocab_size": len(VOCAB), "d_model": 8, "num_labels": 3, "dropout": 0.0}
    if model_type == "tree_transformer":
        return {**common, "nhead": 2, "num_layers": 1, "dim_feedforward": 16,
                "n": 2, "k": 4, "activation": "gelu"}
    if model_type == "transformer":
        return {**common, "nhead": 2, "num_layers": 1, "dim_feedforward": 16,
                "max_seq_len": 32, "activation": "gelu"}
    if model_type == "lstm":
        return {**common, "num_layers": 1}
    return {**common, "hidden_size": 8}


def make_config(root, model_type):
    model = {
        "type": model_type, "d_model": 8, "layers": 1, "heads": 2,
        "dim_feedforward": 16, "max_seq_len": 32, "dropout": 0.0,
        "activation": "gelu", "hidden_size": 8,
    }
    return DotDict({
        "experiment_name": f"tiny_{model_type}",
        "tree": {"branching_factor": 2, "depth": 4},
        "model": model,
        "training": {
            "learning_rate": 0.01, "batch_size": 2, "epochs": 1,
            "weight_decay": 0.0, "log_interval": 10, "n": None,
            "eval_n": None, "seed": 1998, "validation_fraction": 0.25,
            "split_seed": 1998,
        },
        "data": {
            "input_dir": root, "vocab_path": os.path.join(root, "vocab.json"),
            "positions_path": os.path.join(root, "precomputed_positions.pt"),
            "num_workers": 0,
        },
        "paths": {"save_dir": root},
    })


def write_data(root):
    frame = pd.DataFrame({
        "prefix": [["[CLS]", "x"], ["[CLS]", "sin", "x"]],
        "integrand": ["x", "sin(x)"],
        "source": ["elementary", "nonelementary"],
        "label": [[1.0, 2.0, -1.0], [3.0, 1.0, 2.0]],
    })
    frame.to_parquet(os.path.join(root, "train_data.parquet"), index=False)
    frame.to_parquet(os.path.join(root, "test_data.parquet"), index=False)
    with open(os.path.join(root, "vocab.json"), "w") as file:
        json.dump(VOCAB, file)
    torch.save(precompute_all_positions(2, 4), os.path.join(root, "precomputed_positions.pt"))


def make_checkpoint(root, model_type):
    spec = get_model_spec_by_type(model_type)
    params = hyperparameters(model_type)
    model = spec.build_model_from_hyperparameters(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=1)
    cfg = make_config(root, model_type)
    payload = build_v2_checkpoint(
        model=model, optimizer=optimizer, scheduler=scheduler, epoch=1,
        global_step=1, best_val_loss=0.5, epochs_no_improve=0,
        model_type=model_type, model_hyperparameters=params, vocabulary=VOCAB,
        config_snapshot=config_to_dict(cfg),
        data_metadata={
            "batch_size": 2, "training_n": None, "validation_fraction": 0.25,
            "split_seed": 1998, "steps_per_epoch": 1,
        },
        checkpoint_role="best", planned_epochs=1, steps_per_epoch=1,
    )
    path = os.path.join(root, f"{model_type}_best.pth")
    save_checkpoint_payload(path, payload)
    return path, payload


def dgl_available():
    try:
        import dgl  # noqa: F401
        return True
    except ImportError:
        return False


class CheckpointV2Tests(unittest.TestCase):
    def model_types(self):
        types = ["tree_transformer", "transformer", "lstm"]
        if dgl_available():
            types.append("tree_lstm")
        return types

    def test_round_trip_reconstructs_every_available_architecture(self):
        with tempfile.TemporaryDirectory() as root:
            write_data(root)
            for model_type in self.model_types():
                with self.subTest(model_type=model_type):
                    path, original = make_checkpoint(root, model_type)
                    loaded = load_checkpoint_payload(path)
                    spec, model = reconstruct_model_from_checkpoint(loaded)
                    self.assertEqual(spec.model_type, model_type)
                    self.assertEqual(validate_embedded_vocabulary(loaded), VOCAB)
                    self.assertEqual(loaded["vocabulary_sha256"], vocabulary_sha256(VOCAB))
                    model.load_state_dict(checkpoint_state_dict(loaded))
                    self.assertEqual(set(model.state_dict()), set(original["model_state_dict"]))

    def test_typed_and_untyped_legacy_checkpoint_compatibility(self):
        with tempfile.TemporaryDirectory() as root:
            write_data(root)
            for model_type in self.model_types():
                with self.subTest(model_type=model_type):
                    spec = get_model_spec_by_type(model_type)
                    params = hyperparameters(model_type)
                    model = spec.build_model_from_hyperparameters(params)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    typed_path = os.path.join(root, f"typed_{model_type}.pth")
                    save_checkpoint(
                        typed_path, model, optimizer, None, epoch=2,
                        best_val_loss=0.5, epochs_no_improve=1,
                        model_type=model_type, model_hyperparameters=params,
                    )
                    self.assertEqual(
                        load_checkpoint(
                            typed_path, model, expected_model_type=model_type
                        ),
                        (3, 0.5, 1),
                    )
                    with self.assertRaisesRegex(ValueError, "Checkpoint/model mismatch"):
                        load_checkpoint(
                            typed_path,
                            model,
                            expected_model_type=(
                                "transformer"
                                if model_type != "transformer"
                                else "tree_transformer"
                            ),
                        )

                    untyped_path = os.path.join(root, f"untyped_{model_type}.pth")
                    save_checkpoint(
                        untyped_path, model, optimizer, None, epoch=0,
                        best_val_loss=1.0, epochs_no_improve=0,
                    )
                    load_checkpoint(
                        untyped_path, model, expected_model_type=model_type
                    )

    def test_data_parallel_prefix_and_rng_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            write_data(root)
            _, payload = make_checkpoint(root, "lstm")
            payload["model_state_dict"] = {
                f"module.{key}": value for key, value in payload["model_state_dict"].items()
            }
            self.assertTrue(all(not key.startswith("module.") for key in checkpoint_state_dict(payload)))

            state = payload["rng_state"]
            random.random(), np.random.random(), torch.rand(1)
            restore_rng_state(state)
            actual = (random.random(), np.random.random(), torch.rand(1))
            restore_rng_state(state)
            repeated = (random.random(), np.random.random(), torch.rand(1))
            self.assertEqual(actual[0], repeated[0])
            self.assertEqual(actual[1], repeated[1])
            torch.testing.assert_close(actual[2], repeated[2])

    def test_checkpoint_only_inference_writes_aligned_structured_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            write_data(root)
            for model_type in self.model_types():
                with self.subTest(model_type=model_type):
                    path, _ = make_checkpoint(root, model_type)
                    output_dir = os.path.join(root, f"outputs_{model_type}")
                    result = run_inference(
                        path, sample_n=2, device_arg="cpu", num_workers=0,
                        output_dir=output_dir,
                    )
                    self.assertEqual(tuple(result.predictions.shape), (2, 3))
                    prediction_path = os.path.join(
                        output_dir, f"{model_type}_best_test_n2_predictions.parquet"
                    )
                    predictions = pd.read_parquet(prediction_path)
                    self.assertEqual(predictions["row_id"].tolist(), [0, 1])
                    self.assertEqual(predictions["source"].tolist(),
                                     ["elementary", "nonelementary"])
                    self.assertEqual(len(predictions.iloc[0]["predictions"]), 3)
                    for suffix in ("metrics.json", "metrics_by_source.csv",
                                   "metrics_by_complexity.csv"):
                        self.assertTrue(os.path.isfile(os.path.join(
                            output_dir, f"{model_type}_best_test_n2_{suffix}"
                        )))

    def test_exhausted_scheduler_extends_for_every_available_architecture(self):
        with tempfile.TemporaryDirectory() as root:
            write_data(root)
            for model_type in self.model_types():
                with self.subTest(model_type=model_type):
                    _, payload = make_checkpoint(root, model_type)
                    spec = get_model_spec_by_type(model_type)
                    params = hyperparameters(model_type)
                    model = spec.build_model_from_hyperparameters(params)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=0.01, total_steps=2
                    )
                    cfg = make_config(root, model_type)
                    cfg.training.epochs = 2
                    resumed = _resume_training_state(
                        payload, model, optimizer, scheduler, cfg, spec, params,
                        VOCAB, payload["data_metadata"], 1, torch.device("cpu"),
                    )
                    start_epoch, _, _, global_step, new_scheduler, schedule_start = resumed
                    self.assertEqual((start_epoch, global_step, schedule_start), (2, 1, 1))
                    self.assertEqual(new_scheduler.total_steps, 1)

    def test_interrupted_scheduler_restores_and_incompatible_resume_fails(self):
        with tempfile.TemporaryDirectory() as root:
            write_data(root)
            spec = get_model_spec_by_type("lstm")
            params = hyperparameters("lstm")
            cfg = make_config(root, "lstm")
            cfg.training.epochs = 3
            model = spec.build_model_from_hyperparameters(params)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.01, total_steps=3
            )
            optimizer.step()
            scheduler.step()
            metadata = {
                "batch_size": 2, "training_n": None,
                "validation_fraction": 0.25, "split_seed": 1998,
                "steps_per_epoch": 1,
            }
            payload = build_v2_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, epoch=1,
                global_step=1, best_val_loss=0.5, epochs_no_improve=0,
                model_type="lstm", model_hyperparameters=params,
                vocabulary=VOCAB, config_snapshot=config_to_dict(cfg),
                data_metadata=metadata, checkpoint_role="last",
                planned_epochs=3, steps_per_epoch=1,
            )

            resumed_model = spec.build_model_from_hyperparameters(params)
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
            resumed_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                resumed_optimizer, max_lr=0.01, total_steps=3
            )
            resumed = _resume_training_state(
                payload, resumed_model, resumed_optimizer, resumed_scheduler,
                cfg, spec, params, VOCAB, metadata, 1, torch.device("cpu"),
            )
            self.assertIs(resumed[4], resumed_scheduler)
            self.assertEqual(resumed_scheduler.last_epoch,
                             payload["scheduler_state_dict"]["last_epoch"])

            changed = dict(metadata)
            changed["batch_size"] = 4
            with self.assertRaisesRegex(ValueError, "batch_size"):
                _resume_training_state(
                    payload, resumed_model, resumed_optimizer, resumed_scheduler,
                    cfg, spec, params, VOCAB, changed, 1, torch.device("cpu"),
                )

    def test_corrupt_vocabulary_and_missing_checkpoint_fail_clearly(self):
        with tempfile.TemporaryDirectory() as root:
            write_data(root)
            _, payload = make_checkpoint(root, "lstm")
            payload["vocabulary"]["new"] = 99
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_embedded_vocabulary(payload)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                load_checkpoint_payload(os.path.join(root, "missing.pth"))

    def test_structured_metrics_exclude_invalid_labels(self):
        predictions = torch.tensor([[-100.0, 0.1, 0.2], [0.0, 2.0, 1.0]])
        labels = torch.tensor([[-1.0, 1.0, 2.0], [1.0, -1.0, 0.0]])
        masks = labels.ne(-1)
        metrics = compute_evaluation_metrics(predictions, labels, masks)
        self.assertEqual(metrics.evaluated_samples, 2)
        self.assertEqual(metrics.correct_smallest, 1)
        self.assertEqual(metrics.smallest_label_accuracy, 50.0)
        self.assertEqual(metrics.top_2_accuracy, 100.0)
        self.assertTrue(np.isfinite(metrics.rank_loss))
        self.assertTrue(np.isfinite(metrics.masked_mse))

    def test_resume_metrics_are_truncated_or_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "metrics.csv")
            with open(path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=_METRIC_FIELDS)
                writer.writeheader()
                for epoch in (1, 2, 3):
                    writer.writerow({field: epoch for field in _METRIC_FIELDS})
            _prepare_metrics_file(path, resume_epoch=2)
            with open(path, newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["epoch"] for row in rows], ["1", "2"])

            with self.assertRaisesRegex(ValueError, "before checkpoint epoch"):
                _prepare_metrics_file(path, resume_epoch=3)


if __name__ == "__main__":
    unittest.main()
