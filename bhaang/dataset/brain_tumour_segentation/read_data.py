from pycocotools.coco import COCO
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import numpy as np
#import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Read_data():
    """
    Read the data from a specific dataset and convert it to a format suitable for training in pytorch
    """
    def __init__(self):

        self.data_path = '/home/harsh/Lab_Work/Bhaang/bhaang/dataset/brain_tumour_segentation/'
        self.train_path = self.data_path + "train/"
        self.test_path = self.data_path + "test/"
        self.val_path = self.data_path + "valid/"
        self.train_annotation_path = self.train_path + "_annotations.coco.json"
        self.test_annotation_path = self.test_path + "_annotations.coco.json"
        self.val_annotation_path = self.val_path + "_annotations.coco.json"


    def read_annotations(self):
        """
        Read the annotations from the dataset
        """
        self.train_coco = COCO(self.train_annotation_path)
        self.test_coco = COCO(self.test_annotation_path)
        self.val_coco = COCO(self.val_annotation_path)
        
    def load_image_and_mask(self,img_index, split = 'train'):
        """
        Load the image and mask from the dataset
        Args:
            split (str): The split of the dataset. Choose from 'train', 'test', or 'val'.
            img_index (int): The index of the image in the dataset.
        Returns:
            img: np.array: The image from the dataset.
            mask: np.array: The mask from the dataset.
        """
        if split == 'train':
            coco = self.train_coco
            data_path = self.train_path
        elif split == 'test':
            coco = self.test_coco
            data_path = self.test_path
        elif split == 'val':
            coco = self.val_coco
            data_path = self.val_path
        else:
            raise ValueError(f"Invalid split given {split}. Choose from 'train', 'test', or 'val'.")
        
        img_id = coco.getImgIds(imgIds=img_index)
        img_data = coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(data_path, img_data['file_name']))
        ann_ids = coco.getAnnIds(imgIds=img_data['id'])
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((img_data['height'], img_data['width']))
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))
        # add channel dimension (H, W) -> (C, H, W)
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)

        return np.array(img), mask
    
    def __len__(self):
        return len(self.train_coco.getImgIds()), len(self.test_coco.getImgIds()), len(self.val_coco.getImgIds())    

class pytorch_dataset(Dataset):
    def __init__(self, split = 'train', transform=None):
        """
        Initialize the dataset
        Args:
            split (str): The split of the dataset. Choose from 'train', 'test', or 'val'.
            transform (torchvision.transforms): The transform to apply to the image. segementation transforms will be same for the mask.
        """
        # check if the split is valid
        if split not in ['train', 'test', 'val']:
            raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")
        self.split = split
        self.read_data = Read_data()
        self.read_data.read_annotations()
        self.transform = transform
    
    def __len__(self):
        if self.split == 'train':
            return self.read_data.__len__()[0]
        elif self.split == 'test':
            return self.read_data.__len__()[1]
        elif self.split == 'val':
            return self.read_data.__len__()[2]
        else:
            raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")
    
    def __getitem__(self, idx):
        img, mask = self.read_data.load_image_and_mask(idx, self.split)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask
    
# # test the functionallity of the class
# data = Read_data()
# data.read_annotations()
# img, mask = data.load_image_and_mask(0, 'train')
# print(img.shape, mask.shape)
# # use matplotlib to plot the image and mask using subplots
# fig,  ax = plt.subplots(1, 2)
# ax[0].imshow(img[:,:,0], cmap='gray')
# ax[1].imshow(mask, cmap='gray')
# plt.savefig('image_and_mask.png')

# # test the pytorch dataset class
# dataset = pytorch_dataset('train')
# img, mask = dataset[0]
# print(img.shape, mask.shape)
# # use matplotlib to plot the image and mask using subplots
# fig,  ax = plt.subplots(1, 2)
# ax[0].imshow(img[:, :,0], cmap='gray')
# ax[1].imshow(mask, cmap='gray')
# plt.savefig('image_and_mask_using_dataset.png')
