import torch
import torch.nn as nn

class TreeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers=1,
        dim_feedforward=128,
        num_labels=1,
        n=2,
        k=6
    ):
        super(TreeTransformer, self).__init__()
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
            dropout=0.1,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.classifier = nn.Linear(d_model, num_labels)
        self.cls_dropout = nn.Dropout(0.1)
        self.cls_norm    = nn.LayerNorm(d_model)

        # Learnable scaling parameter
        self.p = nn.Parameter(torch.tensor(0.9))

    def forward(self, token_ids, pos_encodings, token_mask, label_mask=None):
        """
        Args:
          token_ids:      LongTensor [B, T]
          pos_encodings:  FloatTensor [B, T, d_model]
          token_mask:     BoolTensor [B, T] (True at padding)
          label_mask:     BoolTensor [B, num_labels] (True for valid labels)
        Returns:
          logits: FloatTensor [B, num_labels]
        """
        # Embedding lookup
        embeddings = self.embedding(token_ids)  # [B, T, d_model]

        # Compute effective p in [-1,1]
        p_eff = torch.tanh(self.p)
        exp_range = torch.arange(0, self.k, device=embeddings.device, dtype=embeddings.dtype)
        p_vals = p_eff ** exp_range
        norm_factor = torch.sqrt(1 - p_eff ** 2)
        scaling_const = torch.sqrt(torch.tensor((self.n * self.k) / 2,
                                                device=embeddings.device,
                                                dtype=embeddings.dtype))
        scaling_vector = p_vals.repeat_interleave(self.n) * norm_factor * scaling_const
        # scaling_vector shape: [d_model]

        # Apply positional encodings
        pos_scaled = pos_encodings * scaling_vector  # [B, T, d_model]
        x = embeddings + pos_scaled                  # [B, T, d_model]

        # Transformer expects shape [T, B, d_model]
        x = x.transpose(0, 1)
        enc_out = self.encoder(x, src_key_padding_mask=token_mask)  # [T, B, d_model]
        enc_out = enc_out.transpose(0, 1)                           # [B, T, d_model]

        # Use the first token (root) for regression
        root = enc_out[:, 0, :]          # [B, d_model]
        root = self.cls_norm(root)
        root = self.cls_dropout(root)
        logits = self.classifier(root)   # [B, num_labels]

        # Zero out missing labels if a mask is provided
        if label_mask is not None:
            logits = logits * label_mask.float()

        return logits
