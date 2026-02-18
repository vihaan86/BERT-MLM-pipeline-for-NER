"""Masked Language Model (MLM) head and BERT for masked word prediction."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import BERTConfig
from .model import BERTModel


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit."""
    return x * 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))


class BertMLMHead(nn.Module):
    """
    MLM head: transform -> GELU -> LayerNorm -> decoder (logits over vocab).
    Optionally ties decoder weight with token embeddings.
    """

    def __init__(self, config: BERTConfig, embedding_layer: Optional[nn.Embedding] = None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.embedding_layer = embedding_layer
        if embedding_layer is not None:
            self.decoder.weight = embedding_layer.weight
        self._init_weights(config.initializer_range)

    def _init_weights(self, std: float):
        self.dense.weight.data.normal_(mean=0.0, std=std)
        self.dense.bias.data.zero_()
        self.layer_norm.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)
        if self.embedding_layer is None:
            self.decoder.weight.data.normal_(mean=0.0, std=std)
        self.decoder.bias.data.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        hidden = self.dense(hidden_states)
        hidden = gelu(hidden)
        hidden = self.layer_norm(hidden)
        return self.decoder(hidden)


class BERTForMaskedLM(nn.Module):
    """
    BERT with a Masked Language Model head for predicting masked tokens.
    Use labels with -100 at non-masked positions so loss is computed only on masked positions.
    """

    def __init__(self, config: BERTConfig, tie_embeddings: bool = True):
        super().__init__()
        self.config = config
        self.bert = BERTModel(config)
        embedding_layer = self.bert.embeddings.token_embeddings if tie_embeddings else None
        self.mlm_head = BertMLMHead(config, embedding_layer=embedding_layer)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Args:
            input_ids: (batch_size, seq_len), use mask_token_id for masked positions
            attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding
            token_type_ids: optional segment ids
            labels: (batch_size, seq_len), true token ids at masked positions, -100 elsewhere
            output_attentions: return attention weights
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar, only if labels is not None (CrossEntropy on masked positions)
            all_attentions: optional
        """
        last_hidden, _, all_attentions = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )
        logits = self.mlm_head(last_hidden)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
        return logits, loss, all_attentions
