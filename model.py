import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, n_clips=4):
        self.n_clips = n_clips

    def get_tube_proposals(self, raw_clips):
        tube_props = np.array([t_prop for t_prop in raw_clips])
        # cartesian product of tube proposals (tubes from one clip do not have connection)
        tube_props = np.array(
            np.meshgrid(tube_props[:, 0], *np.vsplit(tube_props[:, 1:].T, len(tube_props)))).T.reshape(-1, self.n_clips)
        return tube_props

    def link_proposals(self, raw_clips: list):
        """

        :list raw_clips: 8-frame list with proposed tubes per frame
        """
        # all possible combinations of tube proposals (proposals from the same clip are skipped)
        tube_proposals = self.get_tube_proposals(raw_clips)
        scores = []
        for i in range(len(tube_proposals)):
            score = 0
            overlap = 0
            # probability that action is found in the i-th clip
            action_prob = self.compute_actionness(tube_proposals[i][:, -1])
            for j in range(len(tube_proposals[i]) - 1):
                overlap += self.compute_overlap(tube_proposals[i][j, :-1], tube_proposals[i][j + 1, :-1])
            score = self.compute_score(action_prob, overlap)
            scores.append(score)
        best_prop_idx = int(np.argmax(scores))
        best_seq = tube_proposals[best_prop_idx]
        return best_seq

    @staticmethod
    def compute_overlap(tp1, tp2):
        """
            Computes IoU between last and first frames
            in j-th and j+1-th proposals
        """
        xA = max(tp1[0], tp2[0])
        yA = max(tp1[1], tp2[1])
        xB = min(tp1[2], tp2[2])
        yB = min(tp1[3], tp2[3])

        # compute the intersection area
        inters_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if inters_area == 0:
            return 0
        # compute the area of both boxes
        tp1_area = abs((tp1[2] - tp1[0]) * (tp1[3] - tp1[1]))
        tp2_area = abs((tp2[2] - tp2[0]) * (tp2[3] - tp2[1]))
        iou = inters_area / float(tp1_area + tp2_area - inters_area)

        return iou

    @staticmethod
    def compute_actionness(actionness):
        acts = torch.sum(actionness, dim=0)
        return acts

    def compute_score(self, actss, overlap):
        score = 1 / self.n_clips * actss + 1 / (self.n_clips - 1) * overlap
        return score


class ToiPool(nn.Module):
    def __init__(self, d, h, w):
        super(ToiPool, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool3d((d, h, w))

    def forward(self, tube_props):
        return self.pool(tube_props)


class TPN(nn.Module):
    def __init__(self, input_C, fc6_units=8192, fc7_units=4096, fc8_units=4096):
        """Initialize parameters and build model.
        Params
        ======
            fc6_units (int): Number of nodes in first hidden layer
            fc7_units (int): Number of nodes in second hidden layer
        """
        super(TPN, self).__init__()
        self.in_channels = input_C  # output feature map
        self.n_anchor = 9  # no. of anchors at each location
        self.toi2 = ToiPool(8, 8, 8)
        self.toi5 = ToiPool(1, 4, 4)
        self.conv11 = nn.Conv1d(144, 8192, 1)
        self.fc6 = nn.Linear(fc6_units, fc7_units)
        self.fc7 = nn.Linear(fc7_units, fc8_units)
        self.fc8 = nn.Linear(fc8_units, fc8_units)

    def forward(self, bboxes, conv2):
        """

        :list bboxes: conv5 output boxes, each being shaped (x1, y1, x2, y2)
        """
        # Cx8x150x200, Nx19x25
        scaled_bboxes = bboxes.copy()
        scaled_bboxes[:, [0, 2]] *= 150 / 19
        scaled_bboxes[:, [1, 3]] *= 200 / 25
        sliced_conv2 = conv2[:, :, scaled_bboxes[:, 0]:scaled_bboxes[:, 2] + 1,
                             scaled_bboxes[:, 1]:scaled_bboxes[:, 3] + 1]
        x1 = self.toi2(sliced_conv2)
        x1 = torch.norm(x1, p=2)  # 8x8x8
        x2 = self.toi5(bboxes)
        x2 = x2.repeat(8, 1, 1, 1)  # 8x4x4
        x2 = torch.norm(x2, p=2)
        x = torch.cat((x1, x2), dim=1)  # Cx8x12x12
        x = x.reshape((512, 8, -1))  # CxDx144
        reg = self.conv11(torch.flatten(x))
        # C3D pre-trained-model should be used
        reg = self.fc6(reg)
        reg = self.fc7(reg)
        reg = self.fc8(reg)
        return reg


class TCNN(nn.Module):
    """End-to-end action detection model"""

    def __init__(self, input_size, seed, fc8_units=4096):
        """Initialize parameters and build model.
        Params
        ======
            seed (int): Random seed
        """
        super(TCNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.n_anchor = 9  # no. of anchors at each location
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
        self.Linker = Linker()
        self.reg_layer = nn.Conv3d(fc8_units, self.n_anchor * 4, 1, 1, 0)
        self.cls_layer = nn.Conv3d(fc8_units, self.n_anchor * 2, 1, 1, 0)

    def forward(self, x):
        """Build a network that maps anchor boxes to a seq. of frames."""
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(
            self.conv3b(F.leaky_relu(self.conv3a(x)))))
        x = self.pool4(F.leaky_relu(
            self.conv4b(F.leaky_relu(self.conv4a(x)))))
        x = F.leaky_relu(self.conv5b(F.leaky_relu(self.conv5a(x))))
        boxes = self.reg_layer(x)
        action_probs = self.cls_layer(x)
        tube_props = self.TPN(boxes, self.conv2)  # tube proposals for all clips (shape (Nx8xXxY))
        best_t_prop = self.Linker.link_proposals(tube_props)
        return x
