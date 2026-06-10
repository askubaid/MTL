import torch
import torch.nn as nn
from transformers import RobertaModel

class MultiTaskRoBERTa(nn.Module):
    def __init__(self, num_intents=151, num_emotions=28):
        super(MultiTaskRoBERTa, self).__init__()
        # Shared encoder
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Task specific heads
        self.intent_head = nn.Linear(self.roberta.config.hidden_size, num_intents)
        self.emotion_head = nn.Linear(self.roberta.config.hidden_size, num_emotions)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the pooler_output (representation of the [CLS] token)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        intent_logits = self.intent_head(pooled_output)
        emotion_logits = self.emotion_head(pooled_output)
        
        return intent_logits, emotion_logits
