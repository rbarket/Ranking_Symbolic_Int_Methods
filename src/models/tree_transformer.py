import torch
import torch.nn as nn

class TreeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        num_labels: int = 1,
        n: int = 2,
        k: int = 6,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        assert d_model == n * k, "d_model must equal n*k for tree positional encoding."
        self.d_model = d_model
        self.n, self.k = n, k

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.classifier = nn.Linear(d_model, num_labels)
        self.cls_dropout = nn.Dropout(dropout)
        self.cls_norm    = nn.LayerNorm(d_model)

        # Learnable scaling parameter
        self.p = nn.Parameter(torch.tensor(0.9))

    def forward(
        self,
        token_ids: torch.LongTensor,        # [B, T]
        pos_encodings: torch.FloatTensor,   # [B, T, d_model]
        token_mask: torch.BoolTensor,       # [B, T] (True at padding)
        label_mask: torch.BoolTensor | None = None,  # [B, num_labels] (True for valid labels)
        attn_mask: torch.Tensor | None = None,       # optional (T, T) or broadcastable
    ) -> torch.FloatTensor:
        """
        Returns:
          logits: FloatTensor [B, num_labels]
        """
        # Embedding lookup -> [B, T, d_model]
        embeddings = self.embedding(token_ids)

        # Compute effective p in [-1,1]
        p_eff = torch.tanh(self.p)  # scalar
        exp_range = torch.arange(0, self.k, device=embeddings.device, dtype=embeddings.dtype)  # [k]
        p_vals = p_eff ** exp_range                           # [k]
        norm_factor = torch.sqrt(1 - p_eff ** 2 + 1e-12)      # tiny eps for numerical safety

        # scaling_const = sqrt((n*k)/2)
        scaling_const = torch.sqrt(
            torch.tensor((self.n * self.k) / 2.0, device=embeddings.device, dtype=embeddings.dtype)
        )

        # Repeat each p^j "n" times to reach d_model = n*k
        scaling_vector = p_vals.repeat_interleave(self.n) * norm_factor * scaling_const  # [d_model]

        # Apply positional encodings
        pos_scaled = pos_encodings * scaling_vector    # [B, T, d_model]
        x = embeddings + pos_scaled                    # [B, T, d_model]

        enc_out = self.encoder(
            x,
            mask=attn_mask,                             # optional causal or custom attn mask
            src_key_padding_mask=token_mask
        )  # [B, T, d_model]

        # Use the first token (root) for regression
        root = enc_out[:, 0, :]          # [B, d_model]
        root = self.cls_norm(root)
        root = self.cls_dropout(root)
        logits = self.classifier(root)   # [B, num_labels]

        # Zero out missing labels if a mask is provided (expects True for valid)
        if label_mask is not None:
            logits = logits * label_mask.float()

        return logits
