import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

class VideoEncoder(nn.Module):
    def __init__(self, output_dim=768, freeze=True):
        super().__init__()
        
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.projection = nn.Linear(512, output_dim)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def extract_frame(self, video_path, start_time=None, end_time=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if start_time is not None and end_time is not None:
            mid_time = (start_time + end_time) / 2
            target_frame = int(mid_time * fps)
        else:
            target_frame = total_frames // 2
            
        target_frame = min(target_frame, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return torch.zeros(3, 224, 224)
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.projection(features)
        return output

if __name__ == "__main__":
    encoder = VideoEncoder()
    dummy_frames = torch.zeros(2, 3, 224, 224)
    output = encoder(dummy_frames)
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([2, 768])")