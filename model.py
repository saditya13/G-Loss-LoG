"""
Model architectures for text classification.
"""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BertClassifier(nn.Module):
    """BERT-based text classifier."""
    
    def __init__(self, pretrained_model='bert-base-uncased', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = self.bert_model.config.hidden_size
        self.classifier = nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_embeddings = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_embeddings)
        return cls_logit, cls_embeddings