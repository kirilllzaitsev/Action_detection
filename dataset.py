import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
import os
import glob
from pathlib import Path
from skimage import io, transform
from collections import defaultdict

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
        
        self.cat = list(data_map.keys())
        
        self.images = {cat:[file
                for file in data_map[cat]['images']] for cat in self.cat}
        
        self.bbox = {cat:[np.genfromtxt(Path(data_map[list(data_map.keys())[0]]['captions'][0]), dtype=np.uint8)[:-1] 
                for file in data_map[cat]['captions']] for cat in self.cat}
        
        self.transform = transform

    def __len__(self):
        return len(self.cat)*len(self.bbox[self.cat[0]])

    def __getitem__(self, cat, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(self.data_map[cat][index])
        img = img.convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        bbox = torch.from_numpy(self.bbox[cat][index])
    
        sample = {'image': img, 'bbox': bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    
    DATA = Path('data')

    data_map = defaultdict(dict)

    for cat in DATA.iterdir():
        for batch in Path(d).iterdir():
            images = glob.glob(str(batch)+'/*.jpg')
            captions = glob.glob(str(batch)+'/gt/*.txt')
            data_map[str(cat)]['images'] = images
            data_map[str(cat)]['captions'] = captions

    plt.figure()
    plt.imshow(io.imread(data_map[list(data_map.keys())[0]]['images'][0]))
    plt.title('BBox coordinates: '+
            open(data_map[list(data_map.keys())[0]]['captions'][0]).read())
    plt.show()

    dataset = ActionsDataset(data_map)
