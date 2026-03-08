import torch
import torch.nn as nn
import timm

class EfficientNetBaseline(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetBaseline, self).__init__()
        
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove head
            global_pool='avg'
        )
        
        in_features = self.backbone.num_features  # 1280 untuk B0
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out


def get_model(device):
    model = EfficientNetBaseline(num_classes=2, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    return model