"""BERT encoder model: embeddings + stacked transformer encoder."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import BERTConfig
from .embeddings import BERTEmbeddings
from .encoder import BertEncoder


class BERTModel(nn.Module):
    """
    BERT encoder (no task heads).
    Outputs: last hidden state (and optionally pooled [CLS] and attentions).

    Components (for MLM or other tasks):
    - Embeddings: token + position + segment type; LayerNorm + dropout.
    - Encoder: N × [Multi-head self-attention -> Add -> LayerNorm -> FFN -> Add -> LayerNorm].
    - Pooled output: hidden state at [CLS] (first position).
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BertEncoder(config)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Convert (batch_size, seq_len) mask of 0/1 to additive mask for attention.
        0 -> 0.0 (attend), 1 (padding) -> large negative (mask out).
        """
        extended = attention_mask.unsqueeze(1).unsqueeze(2)
        extended = (1.0 - extended) * torch.finfo(dtype).min
        return extended.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[list]]:
        """
        Args:
            input_ids: (batch_size, seq_len) token ids
            attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding
            token_type_ids: (batch_size, seq_len), segment ids
            position_ids: optional position ids
            output_attentions: return attention weights from all layers
        Returns:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            pooled_output: (batch_size, hidden_size) — [CLS] token representation
            all_attentions: optional list of attention tensors
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, embedding_output.dtype
        )
        encoder_outputs, all_attentions = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
        )
        # Pooled output = representation of [CLS] (first token)
        pooled_output = encoder_outputs[:, 0, :]
        return encoder_outputs, pooled_output, all_attentions
