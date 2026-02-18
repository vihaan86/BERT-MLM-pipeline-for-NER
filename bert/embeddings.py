"""BERT embedding layer: token + position + segment type."""

from typing import Optional

import torch
import torch.nn as nn

from .config import BERTConfig


class BERTEmbeddings(nn.Module):
    """Construct token, position, and segment type embeddings."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.max_position_embeddings = config.max_position_embeddings

        self._init_weights(config.initializer_range)

    def _init_weights(self, std: float):
        self.token_embeddings.weight.data.normal_(mean=0.0, std=std)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=std)
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=std)
        self.layer_norm.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len), segment ids (0 or 1)
            position_ids: (batch_size, seq_len), optional
        Returns:
            (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)

        token_emb = self.token_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        segment_emb = self.token_type_embeddings(token_type_ids)

        embeddings = token_emb + position_emb + segment_emb
        return self.dropout(self.layer_norm(embeddings))
