from __future__ import annotations

import torch
import torch.nn as nn


def _load_dgl():
    try:
        import dgl
    except ImportError as exc:
        raise RuntimeError(
            "TreeLSTM requires DGL. Run it in the TreeLSTM_DGL environment."
        ) from exc
    return dgl


class TreeLSTMCell(nn.Module):
    """Binary TreeLSTM cell for DGL child-to-parent message passing."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_iou = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.U_iou = nn.Linear(2 * hidden_size, 3 * hidden_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * hidden_size))
        self.U_f = nn.Linear(2 * hidden_size, 2 * hidden_size)

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        child_h = nodes.mailbox["h"]
        child_c = nodes.mailbox["c"]
        num_children = child_h.size(1)

        if num_children > 2:
            raise ValueError("TreeLSTMCell supports at most two children per node.")

        missing_children = 2 - num_children
        if missing_children:
            hidden_padding = child_h.new_zeros(
                child_h.size(0),
                missing_children,
                self.hidden_size,
            )
            child_h = torch.cat((child_h, hidden_padding), dim=1)
            child_c = torch.cat((child_c, hidden_padding), dim=1)

        concatenated_h = child_h.reshape(child_h.size(0), 2 * self.hidden_size)
        forget_gates = torch.sigmoid(self.U_f(concatenated_h)).reshape(
            child_h.size(0),
            2,
            self.hidden_size,
        )
        reduced_c = torch.sum(forget_gates * child_c, dim=1)

        return {
            "iou": nodes.data["iou"] + self.U_iou(concatenated_h),
            "c": reduced_c,
        }

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        input_gate, output_gate, update = torch.chunk(iou, 3, dim=1)
        input_gate = torch.sigmoid(input_gate)
        output_gate = torch.sigmoid(output_gate)
        update = torch.tanh(update)

        cell = input_gate * update + nodes.data["c"]
        hidden = output_gate * torch.tanh(cell)
        return {"h": hidden, "c": cell}


class TreeLSTM(nn.Module):
    """Two-layer binary TreeLSTM for batched DGL expression trees."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.cell1 = TreeLSTMCell(d_model, hidden_size)
        self.cell2 = TreeLSTMCell(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _validate_graph(self, graph) -> None:
        missing_fields = {
            field for field in ("features", "is_root") if field not in graph.ndata
        }
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"TreeLSTM graph is missing required node fields: {missing}.")

        features = graph.ndata["features"]
        is_root = graph.ndata["is_root"]
        if features.ndim != 1 or features.shape[0] != graph.num_nodes():
            raise ValueError("graph.ndata['features'] must have shape [num_nodes].")
        if is_root.dtype != torch.bool:
            raise ValueError("graph.ndata['is_root'] must have boolean dtype.")
        if is_root.ndim != 1 or is_root.shape[0] != graph.num_nodes():
            raise ValueError("graph.ndata['is_root'] must have shape [num_nodes].")

        if graph.num_nodes() and torch.any(graph.in_degrees() > 2):
            raise ValueError("TreeLSTM supports at most two children per node.")

        offset = 0
        for graph_index, node_count in enumerate(graph.batch_num_nodes().tolist()):
            root_count = is_root[offset : offset + node_count].sum().item()
            if root_count != 1:
                raise ValueError(
                    "Each graph must contain exactly one root; "
                    f"graph {graph_index} contains {root_count}."
                )
            offset += node_count

    def _run_cell(self, graph, cell: TreeLSTMCell, inputs: torch.Tensor) -> torch.Tensor:
        dgl = _load_dgl()
        num_nodes = graph.num_nodes()
        graph.ndata["iou"] = cell.W_iou(inputs)
        graph.ndata["h"] = inputs.new_zeros((num_nodes, self.hidden_size))
        graph.ndata["c"] = inputs.new_zeros((num_nodes, self.hidden_size))
        dgl.prop_nodes_topo(
            graph,
            message_func=cell.message_func,
            reduce_func=cell.reduce_func,
            apply_node_func=cell.apply_node_func,
        )
        return graph.ndata["h"]

    def forward(
        self,
        graph,
        label_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Encode a DGL graph batch and return one score per graph and label.

        Edges must point from children to parents. Token IDs and root markers are
        read from ``graph.ndata['features']`` and ``graph.ndata['is_root']``.
        """
        self._validate_graph(graph)

        with graph.local_scope():
            embeddings = self.dropout(self.embedding(graph.ndata["features"]))
            first_hidden = self._run_cell(graph, self.cell1, embeddings)
            second_inputs = self.dropout(first_hidden)
            second_hidden = self._run_cell(graph, self.cell2, second_inputs)

            root_hidden = self.dropout(second_hidden[graph.ndata["is_root"]])
            logits = self.classifier(root_hidden)

        if label_mask is not None:
            logits = logits * label_mask.to(dtype=logits.dtype)

        return logits
