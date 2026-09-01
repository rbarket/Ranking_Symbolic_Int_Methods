from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """Transformer encoder with sinusoidal positional encodings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        num_labels: int = 1,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1.")

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Precompute the standard sinusoidal positional encodings.
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pos_encoding = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(
            position * div_term[: pos_encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("pos_encoding", pos_encoding)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        encoder_kwargs = {}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            encoder_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            **encoder_kwargs,
        )

        self.classifier = nn.Linear(d_model, num_labels)
        self.cls_dropout = nn.Dropout(dropout)
        self.cls_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_ids: torch.LongTensor,
        pos_encodings: torch.FloatTensor | None,
        token_mask: torch.BoolTensor,
        label_mask: torch.BoolTensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        """
        Encode a padded token batch and return one score per label.

        ``pos_encodings`` is accepted for compatibility with the shared
        TreeTransformer training interface. The Transformer intentionally
        ignores it and uses its internal sinusoidal positional encodings.
        """
        del pos_encodings

        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape [batch, sequence], got {tuple(token_ids.shape)}."
            )

        seq_len = token_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Construct Transformer with a larger max_seq_len."
            )

        embeddings = self.embedding(token_ids) * math.sqrt(self.d_model)
        sinusoidal_positions = self.pos_encoding[:seq_len].to(dtype=embeddings.dtype)
        x = embeddings + sinusoidal_positions.unsqueeze(0)

        enc_out = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=token_mask,
        )

        root = self.cls_norm(enc_out[:, 0, :])
        root = self.cls_dropout(root)
        logits = self.classifier(root)

        if label_mask is not None:
            logits = logits * label_mask.to(dtype=logits.dtype)

        return logits
