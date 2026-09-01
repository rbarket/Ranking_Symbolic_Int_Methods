from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    """Unidirectional LSTM encoder for padded prefix-token sequences."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 1,
        num_labels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.cls_norm = nn.LayerNorm(d_model)
        self.cls_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        token_ids: torch.LongTensor,
        token_mask: torch.BoolTensor,
        label_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Encode each sequence and return one score per label.

        Args:
            token_ids: Token indices with shape ``[batch, sequence]``.
            token_mask: Boolean mask with the same shape; ``True`` marks padding.
            label_mask: Optional ``[batch, num_labels]`` mask for valid labels.
        """
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape [batch, sequence], got {tuple(token_ids.shape)}."
            )
        if token_mask.shape != token_ids.shape:
            raise ValueError(
                "token_mask must have the same shape as token_ids; "
                f"got {tuple(token_mask.shape)} and {tuple(token_ids.shape)}."
            )
        if token_mask.dtype != torch.bool:
            raise ValueError("token_mask must have boolean dtype.")

        lengths = (~token_mask).sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every sequence must contain at least one non-padding token.")

        embeddings = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        root = self.cls_norm(hidden[-1])
        root = self.cls_dropout(root)
        logits = self.classifier(root)

        if label_mask is not None:
            logits = logits * label_mask.to(dtype=logits.dtype)

        return logits
