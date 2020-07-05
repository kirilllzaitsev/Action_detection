import warnings
import glob
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import plot_data_sample, plot_transformed_data_sample

warnings.filterwarnings("ignore")
plt.ion()


class ActionsDataset(Dataset):
    """Actions dataset."""

    def __init__(self, data, transform: transforms.Compose = None):
        """
        Args:
            data (dict): Dictionary with Category - (images, captions) pairs
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.cat = list(map(lambda x: x.strip('data/').lower(), data.keys()))
        self.data = data

        self.images = np.concatenate(
            [data.get(cat).get('images') for cat in data.keys()]).ravel()

        self.bbox = np.concatenate(
            [data.get(cat).get('captions') for cat in data.keys()]).ravel()
        self.bbox = np.array(list(map(lambda x: np.genfromtxt(x, dtype=np.int16)[:-1], self.bbox)))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.images[idx])
        img = img.convert('RGB')

        bbox = self.bbox[idx]
        if self.transform:
            ratio_x, ratio_y = 300 / img.size[0], 400 / img.size[1]
            img = self.transform(img)

            bbox[0] *= ratio_x
            bbox[2] *= ratio_x
            bbox[1] *= ratio_y
            bbox[3] *= ratio_y

        bbox = torch.from_numpy(bbox)
        sample = {'image': img, 'bbox': bbox}

        return sample


if __name__ == "__main__":

    DATA = Path('data')

    data_map = defaultdict(dict)

    for cat in DATA.iterdir():
        if str(cat) == 'data/Lifting':
            continue
        for batch in Path(cat).iterdir():
            images = glob.glob(str(batch) + '/*.jpg')
            captions = glob.glob(str(batch) + '/gt/*.txt')
            if len(captions) < 10:
                continue
            if not data_map.get(str(cat)):
                data_map[str(cat)]['images'] = sorted(images)
                data_map[str(cat)]['captions'] = sorted(captions)
            else:
                data_map[str(cat)]['images'] = data_map[str(cat)]['images'] + sorted(images)
                data_map[str(cat)]['captions'] = data_map[str(cat)]['captions'] + sorted(captions)

    transform = transforms.Compose([
        transforms.Resize((300, 400)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
        transforms.RandomErasing(),
    ])

    dataset = ActionsDataset(data_map, transform)

    plot_data_sample(dataset, 0)
    plot_transformed_data_sample(dataset, 0, transform)
