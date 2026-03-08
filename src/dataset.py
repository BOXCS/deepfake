import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # Load real images (label = 0)
        real_dir = self.data_dir / 'real'
        for img_path in real_dir.glob('*.jpg'):
            self.image_paths.append(img_path)
            self.labels.append(0)
        
        # Load fake images (label = 1)
        fake_dir = self.data_dir / 'fake'
        for img_path in fake_dir.glob('*.jpg'):
            self.image_paths.append(img_path)
            self.labels.append(1)
            
        print(f"[{split}] Real: {self.labels.count(0)} | Fake: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])