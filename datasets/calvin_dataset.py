
import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CalvinDataset(Dataset):
    def __init__(self, data_path=None, image_size=128):

        self.data_path = data_path
        self.data_files = [f for f in os.listdir(data_path) if f.startswith('episode_')]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size)),
            transforms.ToTensor() 
        ])
    
    def __len__(self):
        return len(self.data_files)
    

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        episode_path = os.path.join(self.data_path, file_name)
        data = np.load(episode_path, allow_pickle=True)
        img_static = data["rgb_static"]
        img_static = self.transform(img_static)
        return img_static

class CalvinDatasetSmall(Dataset):
    def __init__(self, data_path=None, caption_path=None, image_size=128):

        self.data_path = data_path
        self.caption_data = self.load_caption_data(caption_path)
        self.data_files = [f for f in os.listdir(data_path) if f.startswith('episode_')]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size)),
            transforms.ToTensor() 
        ])
    def load_caption_data(self, caption_path):
        annotations = np.load(f"{caption_path}", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
        return annotations
        
    def __len__(self):
        return len(self.caption_data)
    

    def __getitem__(self, idx):
        annotation = self.caption_data[idx]
        random_num = random.randint(0, 63)
        start_epi = annotation[0][0]
        epi_num = str(start_epi + random_num).zfill(7)
        file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
        data = np.load(file_path, allow_pickle=True)
        img_static = data["rgb_static"]
        img_static = self.transform(img_static)
        return img_static

