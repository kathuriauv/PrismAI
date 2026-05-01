import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .fusion import FusionLayer

class PrismMasterModel(nn.Module):
    def __init__(self, num_classes=4, num_datasets=2):
        super().__init__()
        # 1. Initialize the Encoders (Frozen by default)
        self.text_encoder = TextEncoder(freeze=True)
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder(freeze=True)
        
        # 2. Initialize the Reliability-Guided Fusion Layer
        self.fusion = FusionLayer(feature_dim=768, num_classes=num_classes, num_datasets=num_datasets)

    def forward(self, input_ids, attention_mask, audio_features, video_frame, dataset_ids):
        # Step 1: Extract features from all three modalities
        text_emb = self.text_encoder(input_ids, attention_mask)
        audio_emb = self.audio_encoder(audio_features)
        video_emb = self.video_encoder(video_frame)
        
        # Step 2: Pass everything to the Fusion Layer
        outputs = self.fusion(text_emb, audio_emb, video_emb, dataset_ids)
        
        # Returns: (logits, text_logits, audio_logits, video_logits, text_unc, audio_unc, video_unc, con_loss)
        return outputs