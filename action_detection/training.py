import glob
import warnings
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from dataset import ActionsDataset
from model import TCNN as Net

warnings.filterwarnings('ignore')

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
            if data_map.get(str(cat)) is None:
                data_map[str(cat)]['images'] = sorted(images)
                data_map[str(cat)]['captions'] = sorted(captions)
            else:
                data_map[str(cat)]['images'] = data_map[str(cat)]['images'] + sorted(images)
                data_map[str(cat)]['captions'] = data_map[str(cat)]['captions'] + sorted(captions)

    input_size = (300, 400, 3)
    SEED = 38

    net = Net(input_size, SEED)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose([
        transforms.Resize((300, 400)),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),

    ])

    dataset = ActionsDataset(data_map, transform)

    BATCH_SIZE = 8
    VAL_SPLIT = .2
    SHUFFLE_DATASET = True
    RANDOM_SEED = 38

    # Creating data indices for training and validation splits:
    DATASET_SIZE = len(dataset)
    indices = list(range(DATASET_SIZE))
    split = int(np.floor(VAL_SPLIT * DATASET_SIZE))
    if SHUFFLE_DATASET:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    sampler=valid_sampler)

    EPOCHS = 10

    for epoch in range(EPOCHS):

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            out = net(torch.transpose(torch.transpose(data['image'].unsqueeze(1), 1, 0), 1, 2))

            loss = criterion(out, data['bbox'])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
