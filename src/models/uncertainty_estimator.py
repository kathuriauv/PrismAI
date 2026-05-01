import torch
import torch.nn as nn

class UncertaintyEstimator(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        uncertainty = self.estimator(features)
        reliability = 1.0 - uncertainty
        return reliability, uncertainty

if __name__ == "__main__":
    estimator = UncertaintyEstimator()
    dummy_features = torch.zeros(2, 768)
    reliability, uncertainty = estimator(dummy_features)
    print(f"Reliability shape: {reliability.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Reliability values (should be between 0 and 1): {reliability}")