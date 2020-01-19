import torch
import torch.nn as nn
import torch.nn.functional as F


class ToiPool(nn.Module):
    def __init__(self, frames, height, width):
        super(ToiPool, self).__init__()
        self.D = frames
        self.height = height
        self.width = width


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
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, (3, 3, 3))
        self.conv3b = nn.Conv3d(256, 256, (3, 3, 3))
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, (3, 3, 3))
        self.conv4b = nn.Conv3d(512, 512, (3, 3, 3))
        self.pool4 = nn.MaxPool3d((2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, (3, 3, 3))
        self.conv5b = nn.Conv3d(512, 512, (3, 3, 3))
        self.toi2 = ToiPool(8, 8, 8)
        self.toi5 = ToiPool(1, 4, 4)
        self.conv11 = nn.Conv1d(512, 8192, 1)
        self.fc6 = nn.Linear(fc6_units, fc7_units)
        self.fc7 = nn.Linear(fc7_units, fc7_units)

    def forward(self, frame):
        """Build a network that maps state -> action values."""
        x = self.pool1(F.leaky_relu(self.conv1(frame)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(
            self.conv3b(F.leaky_relu(self.conv3a(x)))))
        x = self.pool4(F.leaky_relu(
            self.conv4b(F.leaky_relu(self.conv4a(x)))))
        x = self.toi2(F.leaky_relu(
            self.conv5b(F.leaky_relu(self.conv5a(x)))))
        x = self.toi5(x)
        x = self.conv11(x)
        x = self.fc6(x)
        return self.fc7(x)
