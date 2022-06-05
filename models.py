import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.models.deberta.modeling_deberta import ContextPooler


class ToxicRankRoBERTa(nn.Module):
    def __init__(self, model_name):
        super(ToxicRankRoBERTa, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(config.hidden_size, 1)
  
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        classification_output = outputs[1]
        logits = self.classifier(classification_output)
        return logits



class ToxicRankDeBERTa(nn.Module):
    def __init__(self, model_name):
        super(ToxicRankDeBERTa, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(output_dim, 1)
  
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        logits = self.classifier(pooled_output)
        return logits