import os
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

data_path = "/mnt/65af7ccf-e40a-43d3-97d6-f98cc7dd2a74/harshkumar/lung_n_colon/LC25000/lung_colon_image_set"

train_labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
val_labels_df = pd.read_csv(os.path.join(data_path, "val_labels.csv"))
test_labels_df = pd.read_csv(os.path.join(data_path, "test_labels.csv"))

labels = {'colon_n' : 0, 
          'colon_aca' : 1, 
          'lung_n' : 2, 
          'lung_aca' : 3, 
          'lung_scc' : 4}

class pytorch_dataset(Dataset):
    # image_size=(128, 128), as_rgb=True, transform=None, target_transform=None):
    def __init__(self, split="train", image_size=(128, 128), as_rgb=True, transform=None, target_transform=None):
        self.split = split
        if self.split == "train":
            labels_df_path = os.path.join(data_path, "train_labels.csv")
        elif self.split == "val":
            labels_df_path = os.path.join(data_path, "val_labels.csv")
        elif self.split == "test":
            labels_df_path = os.path.join(data_path, "test_labels.csv")
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be one of 'train', 'val', or 'test'.")
        self.labels_df = pd.read_csv(labels_df_path)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif isinstance(image_size, tuple) and len(image_size) == 1:
            image_size = (image_size[0], image_size[0])
        self.image_size = image_size
        self.as_rgb = as_rgb
        resize_transform = [transforms.Resize(self.image_size)]
        
        default_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.as_rgb else transforms.Normalize(mean=[0.5], std=[0.5]),
        ]

        if transform is not None:
            if isinstance(transform, transforms.Compose):
                user_transforms = transform.transforms
            else:
                user_transforms = transform
        else:
            user_transforms = []
        
        all_transforms = resize_transform  + user_transforms + default_transform
        self.transform = transforms.Compose(all_transforms)

        self.target_transform = target_transform
        self.labels = labels
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_path = self.labels_df.iloc[idx]["abs_filepath"]
        label = self.labels_df.iloc[idx]["label"]
        label = self.labels[label]
        image = Image.open(img_path)
        if not self.as_rgb:
            image = image.convert("L")
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        label = torch.tensor(label, dtype=torch.long)
        return image, label
