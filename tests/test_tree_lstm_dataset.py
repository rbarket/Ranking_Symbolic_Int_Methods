import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

try:
    import dgl
except ImportError:
    dgl = None

from src.data.tree_lstm_dataset import (
    TreeLSTMGraphDataset,
    prefix_to_graph_spec,
    tree_lstm_collate_fn,
)


def make_vocab():
    return {
        "<pad>": 0,
        "<OOV>": 1,
        "[CLS]": 2,
        "sub": 3,
        "x": 4,
        "1": 5,
        "sin": 6,
    }


def make_frame():
    return pd.DataFrame(
        {
            "prefix": [
                ["[CLS]", "sub", "x", "1"],
                ["[CLS]", "sin", "x"],
            ],
            "label": [[0.1, -1.0, 0.9], [0.2, 0.4, 0.8]],
            "source": ["elementary", "nonelementary"],
        },
        index=[11, 17],
    )


def make_config():
    return SimpleNamespace(
        model=SimpleNamespace(
            type="tree_lstm",
            d_model=6,
            hidden_size=8,
            dropout=0.0,
        ),
        training=SimpleNamespace(
            batch_size=2,
            validation_fraction=0.1,
            split_seed=1998,
        ),
        data=SimpleNamespace(num_workers=0),
    )


class GraphSpecificationTests(unittest.TestCase):
    def test_binary_graph_has_child_to_parent_edges_and_stable_child_order(self):
        features, sources, destinations, root_id = prefix_to_graph_spec(
            ["[CLS]", "sub", "x", "1"],
            make_vocab(),
        )

        self.assertTrue(torch.equal(features, torch.tensor([2, 3, 4, 5])))
        self.assertTrue(torch.equal(sources, torch.tensor([2, 3, 1])))
        self.assertTrue(torch.equal(destinations, torch.tensor([1, 1, 0])))
        self.assertEqual(root_id, 0)

    def test_unknown_tokens_use_oov_and_malformed_prefix_is_rejected(self):
        features, _, _, _ = prefix_to_graph_spec(
            ["[CLS]", "unknown"],
            make_vocab(),
        )
        self.assertEqual(features.tolist(), [2, 1])

        with self.assertRaisesRegex(ValueError, "Incomplete prefix expression"):
            prefix_to_graph_spec(["[CLS]", "sub", "x"], make_vocab())

    def test_filtering_uses_shared_dataset_behavior_without_loading_dgl(self):
        with patch(
            "src.data.tree_lstm_dataset.load_vocab",
            return_value=make_vocab(),
        ):
            dataset = TreeLSTMGraphDataset(make_config(), dataframe=make_frame())
            self.assertEqual(len(dataset), 2)
            dataset.set_data_type("elementary")
            self.assertEqual(len(dataset), 1)

    def test_missing_dgl_error_is_specific(self):
        with patch(
            "src.data.tree_lstm_dataset.load_vocab",
            return_value=make_vocab(),
        ), patch(
            "src.data.tree_lstm_dataset.import_module",
            side_effect=ImportError("missing dgl"),
        ):
            dataset = TreeLSTMGraphDataset(make_config(), dataframe=make_frame())
            with self.assertRaisesRegex(RuntimeError, "TreeLSTM requires DGL"):
                dataset[0]


@unittest.skipIf(dgl is None, "DGL is not installed in this environment")
class DGLGraphDatasetTests(unittest.TestCase):
    def make_dataset(self):
        with patch(
            "src.data.tree_lstm_dataset.load_vocab",
            return_value=make_vocab(),
        ):
            return TreeLSTMGraphDataset(make_config(), dataframe=make_frame())

    def test_graph_fields_root_and_edge_order(self):
        graph, labels, label_mask = self.make_dataset()[0]
        sources, destinations = graph.edges(order="eid")

        self.assertEqual(graph.num_nodes(), 4)
        self.assertEqual(sources.tolist(), [2, 3, 1])
        self.assertEqual(destinations.tolist(), [1, 1, 0])
        self.assertEqual(graph.ndata["features"].tolist(), [2, 3, 4, 5])
        self.assertEqual(graph.ndata["is_root"].tolist(), [True, False, False, False])
        self.assertEqual(labels.shape, (3,))
        self.assertEqual(label_mask.tolist(), [True, False, True])

    def test_collator_batches_graphs_labels_and_roots(self):
        dataset = self.make_dataset()
        graph, labels, label_mask = tree_lstm_collate_fn(
            [dataset[0], dataset[1]]
        )

        self.assertEqual(graph.batch_size, 2)
        self.assertEqual(graph.batch_num_nodes().tolist(), [4, 3])
        self.assertEqual(graph.ndata["is_root"].sum().item(), 2)
        self.assertEqual(labels.shape, (2, 3))
        self.assertEqual(label_mask.shape, (2, 3))

if __name__ == "__main__":
    unittest.main()
