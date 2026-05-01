import torch
import torch.nn as nn
from transformers import RobertaModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='roberta-base', freeze=True):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.roberta.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

if __name__ == "__main__":
    encoder = TextEncoder()
    dummy_ids = torch.ones(2, 128, dtype=torch.long)
    dummy_mask = torch.ones(2, 128, dtype=torch.long)
    output = encoder(dummy_ids, dummy_mask)
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([2, 768])")