"""BERT model configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BERTConfig:
    """Configuration for BERT encoder architecture."""

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2  # segment A/B
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0

    @property
    def head_dim(self) -> int:
        """Size of each attention head (must divide hidden_size)."""
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads
