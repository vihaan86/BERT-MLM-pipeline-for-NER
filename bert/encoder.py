"""Transformer encoder layer and stack for BERT."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import BERTConfig
from .attention import MultiHeadSelfAttention


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit."""
    return x * 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))


class BertEncoderLayer(nn.Module):
    """
    Single BERT encoder block. Each sublayer is: output = LayerNorm(x + Sublayer(x)).
    - Sublayer 1: Multi-head self-attention (then Add + LayerNorm).
    - Sublayer 2: Feed-forward (linear -> GELU -> linear) (then Add + LayerNorm).
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = gelu

        self._init_weights(config.initializer_range)

    def _init_weights(self, std: float):
        self.intermediate.weight.data.normal_(mean=0.0, std=std)
        self.intermediate.bias.data.zero_()
        self.output.weight.data.normal_(mean=0.0, std=std)
        self.output.bias.data.zero_()
        self.attention_layer_norm.bias.data.zero_()
        self.attention_layer_norm.weight.data.fill_(1.0)
        self.ffn_layer_norm.bias.data.zero_()
        self.ffn_layer_norm.weight.data.fill_(1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: optional attention mask
        Returns:
            layer_output: (batch_size, seq_len, hidden_size)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # Self-attention sublayer: Add (residual) + LayerNorm
        attn_output, attention_weights = self.attention(hidden_states, attention_mask)
        hidden_states = hidden_states + attn_output
        hidden_states = self.attention_layer_norm(hidden_states)

        # Feed-forward sublayer: Add (residual) + LayerNorm
        intermediate = self.act(self.intermediate(hidden_states))
        ffn_output = self.output(intermediate)
        hidden_states = hidden_states + self.dropout(ffn_output)
        layer_output = self.ffn_layer_norm(hidden_states)
        return layer_output, attention_weights


class BertEncoder(nn.Module):
    """Stack of BERT encoder layers."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.layers = nn.ModuleList([BertEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: optional
            output_attentions: whether to return all layer attention weights
        Returns:
            last_hidden_state: (batch_size, seq_len, hidden_size)
            all_attentions: list of (batch_size, num_heads, seq_len, seq_len) or None
        """
        all_attentions = [] if output_attentions else None
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, attention_mask)
            if output_attentions:
                all_attentions.append(attn_weights)
        return hidden_states, all_attentions
