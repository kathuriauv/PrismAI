import torch
import torch.nn as nn
import torch.nn.functional as F
from .uncertainty_estimator import UncertaintyEstimator

class FusionLayer(nn.Module):
    def __init__(self, feature_dim=768, num_classes=4, num_datasets=2):
        super().__init__()
        self.text_uncertainty = UncertaintyEstimator(feature_dim)
        self.audio_uncertainty = UncertaintyEstimator(feature_dim)
        self.video_uncertainty = UncertaintyEstimator(feature_dim)
        
        self.text_classifier = nn.Linear(feature_dim, num_classes)
        self.audio_classifier = nn.Linear(feature_dim, num_classes)
        self.video_classifier = nn.Linear(feature_dim, num_classes)
        
        # dataset calibration: 0=IEMOCAP, 1=MELD
        self.dataset_embedding = nn.Embedding(num_datasets, feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def consistency_loss(self, text_logits, audio_logits, video_logits):
        text_probs = F.softmax(text_logits, dim=1)
        audio_probs = F.softmax(audio_logits, dim=1)
        video_probs = F.softmax(video_logits, dim=1)
        
        loss_ta = F.kl_div(text_probs.log(), audio_probs, reduction='batchmean')
        loss_tv = F.kl_div(text_probs.log(), video_probs, reduction='batchmean')
        loss_av = F.kl_div(audio_probs.log(), video_probs, reduction='batchmean')
        
        return (loss_ta + loss_tv + loss_av) / 3.0

    def forward(self, text_features, audio_features, video_features, dataset_ids):
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
        
        # add dataset-specific calibration vector
        dataset_bias = self.dataset_embedding(dataset_ids)
        fused = fused + dataset_bias
        
        text_logits = self.text_classifier(text_features)
        audio_logits = self.audio_classifier(audio_features)
        video_logits = self.video_classifier(video_features)
        
        logits = self.classifier(fused)
        
        con_loss = self.consistency_loss(text_logits, audio_logits, video_logits)
        
        return (logits, text_logits, audio_logits, video_logits,
                text_uncertainty, audio_uncertainty, video_uncertainty,
                con_loss)

if __name__ == "__main__":
    fusion = FusionLayer()
    dummy_text = torch.zeros(2, 768)
    dummy_audio = torch.zeros(2, 768)
    dummy_video = torch.zeros(2, 768)
    # 0 = IEMOCAP, 1 = MELD
    dummy_dataset_ids = torch.tensor([0, 1])
    
    (logits, text_logits, audio_logits, video_logits,
     text_unc, audio_unc, video_unc, con_loss) = fusion(
        dummy_text, dummy_audio, dummy_video, dummy_dataset_ids)
    
    text_rel = 1.0 - text_unc
    audio_rel = 1.0 - audio_unc
    video_rel = 1.0 - video_unc
    total = text_rel + audio_rel + video_rel
    
    text_w = text_rel / total
    audio_w = audio_rel / total
    video_w = video_rel / total
    
    print(f"Logits shape: {logits.shape}")
    print(f"Consistency loss: {con_loss.item():.4f}")
    print(f"Weights sum (should be 1.0): {(text_w + audio_w + video_w).mean().item():.4f}")
    print(f"Dataset embedding shape: {fusion.dataset_embedding.weight.shape}")