from __future__ import annotations

from importlib import import_module

import torch

from src.data.dataset import FilteredPrefixDataset
from src.utils.io import load_vocab
from src.utils.tree_utils import prefix_to_tree


def _load_dgl():
    try:
        return import_module("dgl")
    except ImportError as exc:
        raise RuntimeError(
            "TreeLSTM requires DGL. Run it in the TreeLSTM_DGL environment."
        ) from exc


def prefix_to_graph_spec(tokens, vocab):
    """Convert prefix tokens to deterministic child-to-parent graph tensors."""
    tree = prefix_to_tree(tokens)
    features = []
    sources = []
    destinations = []
    oov_id = vocab["<OOV>"]

    def visit(node):
        node_id = len(features)
        features.append(vocab.get(node.value, oov_id))

        # Traversing left before right makes incoming edge IDs encode child order.
        for child in (node.left, node.right):
            if child is not None:
                child_id = visit(child)
                sources.append(child_id)
                destinations.append(node_id)
        return node_id

    root_id = visit(tree)
    return (
        torch.tensor(features, dtype=torch.long),
        torch.tensor(sources, dtype=torch.int64),
        torch.tensor(destinations, dtype=torch.int64),
        root_id,
    )


class TreeLSTMGraphDataset(FilteredPrefixDataset):
    """Airy prefix expressions represented as binary DGL graphs."""

    def _build_data(self, dataframe):
        self.vocab = load_vocab(self.cfg)
        self.data = list(
            zip(
                dataframe.index.tolist(),
                dataframe["prefix"].tolist(),
                dataframe["label"].tolist(),
            )
        )

    def __getitem__(self, index):
        row_index, tokens, labels = self.data[index]
        try:
            features, sources, destinations, root_id = prefix_to_graph_spec(
                tokens,
                self.vocab,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid prefix expression at parquet row {row_index}: {exc}"
            ) from exc

        dgl = _load_dgl()
        graph = dgl.graph(
            (sources, destinations),
            num_nodes=features.numel(),
        )
        graph.ndata["features"] = features
        graph.ndata["is_root"] = torch.zeros(
            features.numel(),
            dtype=torch.bool,
        )
        graph.ndata["is_root"][root_id] = True

        label_tensor = torch.tensor(labels, dtype=torch.float32)
        label_mask = label_tensor.ne(-1)
        return graph, label_tensor, label_mask


def tree_lstm_collate_fn(batch):
    dgl = _load_dgl()
    graphs, labels, label_masks = zip(*batch)
    return (
        dgl.batch(graphs),
        torch.stack(labels),
        torch.stack(label_masks),
    )
