import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
import os
import glob
from pathlib import Path
from skimage import io, transform
from collections import defaultdict
from utils import plot_data_sample, plot_transformed_data_sample

warnings.filterwarnings("ignore")
plt.ion()


class ActionsDataset(Dataset):
    """Actions dataset."""

    def __init__(self, data_map, transform=None):
        """
        Args:
            data_map (dict): Dictionary with Category - (images, captions) pairs
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.cat = list(map(lambda x: x.strip('data/').lower(), data_map.keys()))
        self.data_map = data_map
        
        self.images = {cat.strip('data/').lower():[file
                for file in data_map.get(cat).get('images')] for cat in data_map.keys()}
        
        self.bbox = {cat.strip('data/').lower():[np.genfromtxt(file, dtype=np.int16)[:-1] 
                for file in data_map.get(cat).get('captions')] for cat in data_map.keys()}
        
        self.transform = transform

    def __len__(self):
        return len(self.cat)*len(self.bbox[self.cat[0]])

    def __getitem__(self, cat, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.images[cat][idx])
        img = img.convert('RGB')
            
        bbox = torch.from_numpy(self.bbox[cat][idx])
       
        if self.transform:
            img = self.transform(img)
            ratio_x, ratio_y = 300 / img.shape[0], 300 / img.shape[1]
            bbox[0, 2] *= ratio_x
            bbox[1, 3] *= ratio_y

        sample = {'image': img, 'bbox': bbox}

        return sample


if __name__ == "__main__":
    
    DATA = Path('data')

    data_map = defaultdict(dict)

    for cat in DATA.iterdir():
        for batch in Path(cat).iterdir():
            images = glob.glob(str(batch)+'/*.jpg')
            captions = glob.glob(str(batch)+'/gt/*.txt')
            data_map[str(cat)]['images'] = images
            data_map[str(cat)]['captions'] = captions

  
    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
        transforms.RandomErasing(),
    ])
    
    dataset = ActionsDataset(data_map, transform)
    
    plot_data_sample(dataset, 'diving-side', 0)
    plot_transformed_data_sample(dataset, 'diving-side', 0, transform)
    
    

