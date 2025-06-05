from torch.utils.data import Dataset
import medmnist
from medmnist import INFO
import os
import numpy as np
import torch
#from matplotlib import pyplot as plt



class pytorch_dataset(Dataset):
    def __init__(self, split = 'train', data_flag = "organamnist", download=True, as_rgb=False, size=28, transform=None, target_transform=None):
        """
        Initialize the dataset
        
        """
        # check if the split is valid
        if split not in ['train', 'test', 'val']:
            raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")
        self.split = split
        self.data_flag = data_flag
        self.download_flag = download
        self.as_rgb = as_rgb
        self.info = INFO[data_flag]
        self.size = size
        self.size_flag = f"_{size}"
        if size == 28:
            self.size_flag = ""
        if size not in [28, 64, 128, 224]:
            raise ValueError("Invalid size. Choose from 28, 64, 128 or 224.")
        self.root = "/home/harsh/Lab_Work/dataset/medmnist/"

        if download:
            self.download()
        
        npz_file = np.load(os.path.join(self.root, f"{self.data_flag}{self.size_flag}.npz"))
        
        self.imgs = npz_file[f"{self.split}_images"]
        self.labels = npz_file[f"{self.split}_labels"]

        self.transform = transform
        self.target_transform = target_transform
        

        

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url=self.info[f"url{self.size_flag}"],
                root=self.root,
                filename=f"{self.data_flag}{self.size_flag}.npz",
                md5=self.info[f"MD5{self.size_flag}"],
            )
        except:
            raise RuntimeError(
                f"""
                Automatic download failed! Please download {self.data_flag}{self.size_flag}.npz manually.
                1. [Optional] Check your network connection: 
                2. Download the npz file from the Zenodo repository or its Zenodo data link: 
                    {self.info[f"url{self.size_flag}"]}
                3. [Optional] Verify the MD5: 
                    {self.info[f"MD5{self.size_flag}"]}
                4. Put the npz file under your MedMNIST root folder: 
                    {self.root}
                """
            )
    def save(self, folder, postfix="png", write_csv=True):
        from medmnist.utils import save2d

        save2d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, f"{self.data_flag}{self.size_flag}"),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.data_flag}{self.size_flag}.csv")
            if write_csv
            else None,
        )

    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(
            imgs=self.imgs, n_channels=self.info["n_channels"], sel=sel
        )

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(
                os.path.join(
                    save_folder, f"{self.flag}{self.size_flag}_{self.split}_montage.jpg"
                )
            )

        return montage_img
    
    def __len__(self):
        assert self.info["n_samples"][self.split] == self.imgs.shape[0]
        return self.imgs.shape[0]
    
    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} of size {self.size} ({self.data_flag}{self.size_flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)

        if len(img.shape) == 3:
            if img.shape[0] == 3:
                pass
            if img.shape[-1] == 3:  
                img = np.transpose(img, (2, 0, 1))
            else:  
                if self.as_rgb:
                    img = np.stack([img]*3, axis=0 if img.shape[0] != 3 else -1)
        elif len(img.shape) == 2:  
            if self.as_rgb:
                img = np.stack([img]*3, axis=0) 
            else:
                img = np.expand_dims(img, axis=0)  # Add channel dimension

        # convert the target to a tensor
        target = torch.tensor(target, dtype=torch.long)
        img = torch.tensor(np.array(img) / 255, dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)        

        return img, target

# test the dataset
# train_dataset = pytorch_dataset("train", "organamnist", download=False, as_rgb=False, size=28)
# print(train_dataset)
# #train_dataset.save("/home/harsh/Lab_Work/dataset/medmnist/TRAIN/", postfix="png", write_csv=True)

# #get a single image and label
# img, label = train_dataset[0]
# # display the image
# plt.imshow(img)
# plt.xlabel(label)
# plt.show()
