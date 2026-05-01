import torch
import torch.nn as nn
from .uncertainty_estimator import UncertaintyEstimator

class FusionLayer(nn.Module):
    def __init__(self, feature_dim=768, num_classes=4):
        super().__init__()
        self.text_uncertainty = UncertaintyEstimator(feature_dim)
        self.audio_uncertainty = UncertaintyEstimator(feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, text_features, audio_features):
        text_reliability, text_uncertainty = self.text_uncertainty(text_features)
        audio_reliability, audio_uncertainty = self.audio_uncertainty(audio_features)
        
        total_reliability = text_reliability + audio_reliability
        text_weight = text_reliability / total_reliability
        audio_weight = audio_reliability / total_reliability
        
        fused = text_weight * text_features + audio_weight * audio_features
        
        logits = self.classifier(fused)
        
        return logits, text_uncertainty, audio_uncertainty

if __name__ == "__main__":
    fusion = FusionLayer()
    dummy_text = torch.zeros(2, 768)
    dummy_audio = torch.zeros(2, 768)
    logits, text_unc, audio_unc = fusion(dummy_text, dummy_audio)
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: torch.Size([2, 4])")
    print(f"Text uncertainty: {text_unc}")
    print(f"Audio uncertainty: {audio_unc}")