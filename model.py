import torch
import torch.nn as nn
import torch.nn.functional as F


class ToiPool():
    def __init__(self):
        pass

    
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, seed, fc6_units=8192, fc7_units=4096):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(input_size, 64, (3, 3, 3))
        self.pool1 = nn.MaxPool3d((1,2,2))
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3))
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3))
        self.conv4 = nn.Conv3d(256, 512, (3, 3, 3))
        self.conv5 = nn.Conv3d(512, 512, (3, 3, 3))
        self.toi2 = ToiPool(512, (8, 8, 8))
        self.toi5 = ToiPool(128, (8, 8, 8))
        self.conv11 = nn.Conv1d(512)
        self.fc6 = nn.Linear(fc6_units, fc7_units)
        self.fc7 = nn.Linear(fc7_units, fc7_units)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)