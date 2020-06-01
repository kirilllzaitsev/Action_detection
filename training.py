import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from dataset import ActionsDataset
from model import TCNN as Net
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    
    DATA = Path('data')

    data_map = defaultdict(dict)

    for cat in DATA.iterdir():
        for batch in Path(cat).iterdir():
            images = glob.glob(str(batch)+'/*.jpg')
            captions = glob.glob(str(batch)+'/gt/*.txt')
            if data_map.get(str(cat)) is None:
                data_map[str(cat)]['images'] = sorted(images)
                data_map[str(cat)]['captions'] = sorted(captions)
            else:
                data_map[str(cat)]['images'] = data_map[str(cat)]['images']+sorted(images)
                data_map[str(cat)]['captions'] = data_map[str(cat)]['captions']+sorted(captions)
            
    input_size = (300, 300, 3)
    seed = 38

    net = Net(input_size, seed)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    data = ActionsDataset(data_map)
    train_data = None; test_data = None

    EPOCHS = 10

    for epoch in range(EPOCHS):

        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            inputs, labels = data

            optimizer.zero_grad()
            out = net(inputs)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i%2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                         (epoch + 1, i+1, running_loss / 2000))
                running_loss = 0.0