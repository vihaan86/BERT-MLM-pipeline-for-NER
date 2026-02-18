"""Token classification (NER) head and BERT for named entity recognition."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import BERTConfig
from .model import BERTModel


class BertTokenClassificationHead(nn.Module):
    """Linear classifier on top of each token representation for NER/token classification."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, seq_len, num_labels)
        """
        hidden = self.dropout(hidden_states)
        return self.classifier(hidden)


class BERTForTokenClassification(nn.Module):
    """
    BERT with a token classification head (e.g. for NER).
    Labels shape = input_ids; use -100 at positions to ignore (e.g. padding, continuation subwords).
    """

    def __init__(self, config: BERTConfig, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.bert = BERTModel(config)
        self.classifier = BertTokenClassificationHead(
            config.hidden_size, num_labels, dropout_prob=dropout_prob
        )

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
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len), 1 for real tokens, 0 for padding
            token_type_ids: optional segment ids
            labels: (batch_size, seq_len), label id per token, -100 to ignore
            output_attentions: return attention weights
        Returns:
            logits: (batch_size, seq_len, num_labels)
            loss: scalar if labels provided
            all_attentions: optional
        """
        last_hidden, _, all_attentions = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )
        logits = self.classifier(last_hidden)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )
        return logits, loss, all_attentions
