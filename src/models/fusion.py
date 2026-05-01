import torch
import torch.nn as nn
from .uncertainty_estimator import UncertaintyEstimator

class FusionLayer(nn.Module):
    def __init__(self, feature_dim=768, num_classes=4):
        super().__init__()
        self.text_uncertainty = UncertaintyEstimator(feature_dim)
        self.audio_uncertainty = UncertaintyEstimator(feature_dim)
        self.video_uncertainty = UncertaintyEstimator(feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, text_features, audio_features, video_features):
        text_reliability, text_uncertainty = self.text_uncertainty(text_features)
        audio_reliability, audio_uncertainty = self.audio_uncertainty(audio_features)
        video_reliability, video_uncertainty = self.video_uncertainty(video_features)
        
        total_reliability = text_reliability + audio_reliability + video_reliability
        text_weight = text_reliability / total_reliability
        audio_weight = audio_reliability / total_reliability
        video_weight = video_reliability / total_reliability
        
        fused = (text_weight * text_features + 
                 audio_weight * audio_features + 
                 video_weight * video_features)
        
        logits = self.classifier(fused)
        
        return logits, text_uncertainty, audio_uncertainty, video_uncertainty

if __name__ == "__main__":
    fusion = FusionLayer()
    dummy_text = torch.zeros(2, 768)
    dummy_audio = torch.zeros(2, 768)
    dummy_video = torch.zeros(2, 768)
    logits, text_unc, audio_unc, video_unc = fusion(dummy_text, dummy_audio, dummy_video)
    
    # recalculate weights the same way forward() does
    text_rel = 1.0 - text_unc
    audio_rel = 1.0 - audio_unc
    video_rel = 1.0 - video_unc
    total = text_rel + audio_rel + video_rel
    
    text_w = text_rel / total
    audio_w = audio_rel / total
    video_w = video_rel / total
    
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: torch.Size([2, 4])")
    print(f"Text uncertainty: {text_unc.mean().item():.4f}")
    print(f"Audio uncertainty: {audio_unc.mean().item():.4f}")
    print(f"Video uncertainty: {video_unc.mean().item():.4f}")
    print(f"Text weight: {text_w.mean().item():.4f}")
    print(f"Audio weight: {audio_w.mean().item():.4f}")
    print(f"Video weight: {video_w.mean().item():.4f}")
    print(f"Weights sum (should be 1.0): {(text_w + audio_w + video_w).mean().item():.4f}")