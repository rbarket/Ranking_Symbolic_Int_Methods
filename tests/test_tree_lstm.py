import unittest
from types import SimpleNamespace

import torch

try:
    import dgl
except ImportError:
    dgl = None

from src.models.tree_lstm import TreeLSTM, TreeLSTMCell


def make_graph(edges, features, root_index):
    sources, destinations = edges
    graph = dgl.graph((sources, destinations), num_nodes=len(features))
    graph.ndata["features"] = torch.tensor(features, dtype=torch.long)
    graph.ndata["is_root"] = torch.zeros(len(features), dtype=torch.bool)
    graph.ndata["is_root"][root_index] = True
    return graph


@unittest.skipIf(dgl is None, "DGL is not installed in this environment")
class TreeLSTMTests(unittest.TestCase):
    def make_model(self, **overrides):
        kwargs = {
            "vocab_size": 20,
            "d_model": 6,
            "hidden_size": 8,
            "num_labels": 3,
            "dropout": 0.0,
        }
        kwargs.update(overrides)
        return TreeLSTM(**kwargs)

    def graph_examples(self):
        return {
            "single": make_graph(([], []), [2], 0),
            "unary": make_graph(([0], [1]), [2, 3], 1),
            "binary": make_graph(([0, 1], [2, 2]), [2, 3, 4], 2),
        }

    def test_single_unary_binary_and_batched_output_shapes(self):
        model = self.make_model().eval()
        examples = self.graph_examples()

        with torch.no_grad():
            for name, graph in examples.items():
                with self.subTest(name=name):
                    self.assertEqual(model(graph).shape, (1, 3))

            batched = dgl.batch(list(examples.values()))
            self.assertEqual(model(batched).shape, (3, 3))

    def test_unary_reduction_zero_pads_missing_child(self):
        cell = TreeLSTMCell(input_size=1, hidden_size=1)
        with torch.no_grad():
            cell.U_iou.weight.zero_()
            cell.U_f.weight.zero_()
            cell.U_f.bias.zero_()

        nodes = SimpleNamespace(
            mailbox={
                "h": torch.zeros(1, 1, 1),
                "c": torch.tensor([[[2.0]]]),
            },
            data={"iou": torch.zeros(1, 3)},
        )

        reduced = cell.reduce_func(nodes)
        torch.testing.assert_close(reduced["c"], torch.tensor([[1.0]]))

    def test_label_mask_and_backpropagation(self):
        graph = dgl.batch(
            [
                make_graph(([0], [1]), [2, 3], 1),
                make_graph(([0, 1], [2, 2]), [4, 5, 6], 2),
            ]
        )
        model = self.make_model()
        label_mask = torch.tensor([[True, False, True], [False, True, True]])

        output = model(graph, label_mask)

        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.equal(output[~label_mask], torch.zeros(2)))
        output.sum().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_forward_is_repeatable_and_does_not_leak_node_state(self):
        graph = make_graph(([0, 1], [2, 2]), [2, 3, 4], 2)
        original_fields = set(graph.ndata.keys())
        original_features = graph.ndata["features"].clone()
        original_roots = graph.ndata["is_root"].clone()
        model = self.make_model().eval()

        with torch.no_grad():
            first = model(graph)
            second = model(graph)

        torch.testing.assert_close(first, second)
        self.assertEqual(set(graph.ndata.keys()), original_fields)
        self.assertTrue(torch.equal(graph.ndata["features"], original_features))
        self.assertTrue(torch.equal(graph.ndata["is_root"], original_roots))

    def test_output_uses_graph_and_model_device(self):
        graph = make_graph(([0], [1]), [2, 3], 1)
        model = self.make_model()

        output = model(graph)

        self.assertEqual(output.device, model.embedding.weight.device)
        self.assertEqual(output.device, graph.ndata["features"].device)

    def test_rejects_missing_graph_fields(self):
        model = self.make_model()
        graph = dgl.graph(([], []), num_nodes=1)

        with self.assertRaisesRegex(ValueError, "features, is_root"):
            model(graph)

    def test_rejects_multiple_roots(self):
        model = self.make_model()
        graph = make_graph(([0], [1]), [2, 3], 1)
        graph.ndata["is_root"][:] = True

        with self.assertRaisesRegex(ValueError, "exactly one root"):
            model(graph)

    def test_rejects_nodes_with_more_than_two_children(self):
        model = self.make_model()
        graph = make_graph(([0, 1, 2], [3, 3, 3]), [2, 3, 4, 5], 3)

        with self.assertRaisesRegex(ValueError, "at most two children"):
            model(graph)


if __name__ == "__main__":
    unittest.main()
