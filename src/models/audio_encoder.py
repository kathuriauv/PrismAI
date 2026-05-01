import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.projection = nn.Linear(128 * 4 * 4, output_dim)

    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        output = self.projection(features)
        return output

if __name__ == "__main__":
    encoder = AudioEncoder()
    dummy_audio = torch.zeros(2, 1, 64, 178)
    output = encoder(dummy_audio)
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([2, 768])")