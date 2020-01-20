import torch
import torch.nn as nn
import torch.nn.functional as F


class _MaxPoolNd(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class Linker:
    def __init__(self):
        pass


class ToiPool(_MaxPoolNd):
    def __init__(self, kernel_size, d, h, w):
        super(ToiPool, self).__init__(kernel_size)
        self.D = d
        self.height = h
        self.width = w

    def forward(self, frame):
        return F.max_pool2d(frame, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)


class TPN(nn.Module):
    def __init__(self, input_C, fc6_units=8192, fc7_units=4096):
        super(TPN, self).__init__()
        self.in_channels = input_C  # output feature map
        self.mid_channels = 4096
        self.n_anchor = 9  # no. of anchors at each location
        self.toi2 = ToiPool(3, 8, 8, 8)
        self.toi5 = ToiPool(3, 1, 4, 4)
        self.conv11 = nn.Conv1d(512, 8192, 1)
        self.fc6 = nn.Linear(fc6_units, fc7_units)
        self.fc7 = nn.Linear(fc7_units, fc7_units)
        self.reg_layer = nn.Conv3d(self.mid_channels, self.n_anchor * 4, 1, 1, 0)
        self.cls_layer = nn.Conv3d(self.mid_channels, self.n_anchor * 2, 1, 1, 0)

    def forward(self, x1, x2):
        x = self.toi2(x1)
        x = self.toi5(x2)
        x = self.conv11(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return self.fc7(x)


class TCNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(TCNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(input_size, 64, (3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, (3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, (3, 3, 3), padding=1)
        self.pool4 = nn.MaxPool3d((2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, (3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, (3, 3, 3), padding=1)
        self.TPN = TPN(512)


    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(
            self.conv3b(F.leaky_relu(self.conv3a(x)))))
        x = self.pool4(F.leaky_relu(
            self.conv4b(F.leaky_relu(self.conv4a(x)))))
        x = F.leaky_relu(self.conv5b(F.leaky_relu(self.conv5a(x))))
        reg, clf = self.TPN(x)

        return x
