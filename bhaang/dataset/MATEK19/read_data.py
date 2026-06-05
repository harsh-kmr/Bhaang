import os
import pandas as pd
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset


csv_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "Matek-19 Dataset",
    "matek19.csv",
)

label_to_index = {'LYT': 0,
  'NGS': 1,
  'MON': 2,
  'MYO': 3,
  'EOS': 4,
  'MYB': 5,
  'BAS': 6,
  'PMO': 7,
  'NGB': 8,
  'MOB': 9,
  'EBO': 10,
  'LYA': 11,
  'KSC': 12,
  'PMB': 13,
  'MMZ': 14}
index_to_label = {v: k for k, v in label_to_index.items()}

class pytorch_dataset(Dataset):
    def __init__(self, split="train", image_size=(128, 128), as_rgb=True, transform=None, target_transform=None):
        self.split = split
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif isinstance(image_size, tuple) and len(image_size) == 1:
            image_size = (image_size[0], image_size[0])
        self.image_size = image_size
        self.as_rgb = as_rgb
        self.data = pd.read_csv(csv_path)
        
        self.data = self.data[self.data['split'] == self.split].reset_index(drop=True)

        resize_transform = [torchvision.transforms.Resize(self.image_size)]
        
        default_transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.as_rgb else torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]

        if transform is not None:
            if isinstance(transform, torchvision.transforms.Compose):
                user_transforms = transform.transforms
            else:
                user_transforms = transform
        else:
            user_transforms = []
        
        all_transforms = resize_transform  + user_transforms + default_transform
        self.transform = torchvision.transforms.Compose(all_transforms)

        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['converted_abs_path']
        label = row['label']
        image = Image.open(img_path).convert("RGB" if self.as_rgb else "L")
        label = torch.tensor(label_to_index[label])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label