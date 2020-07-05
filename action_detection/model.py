import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_CLASSES = 11  # actions + background


class Linker:
    """
    Linker object matches most relevant tube proposals
    from a sequence of clips

    """
    def __init__(self, n_clips=4):
        self.n_clips = n_clips

    def make_tube_proposals(self, raw_clips: torch.tensor) -> np.array:
        """
        Cartesian product of proposed tubes to find best pairings

        """
        tube_props = raw_clips.numpy()
        # cartesian product of tube proposals (tubes from one clip do not have connection)
        tube_props = np.meshgrid(tube_props[:, 0],
                                 *np.vsplit(tube_props[:, 1:].T, len(tube_props))) \
            .T.reshape(-1, self.n_clips)
        return tube_props

    def link_proposals(self, raw_clips: torch.tensor) -> np.array:
        """
        Compute tube prop. scores and return best sequence of tube proposals

        :list raw_clips: 8-frame list with proposed tubes per frame
        """
        # possible combinations of tube proposals (proposals from the same clip are skipped)
        tube_proposals = self.make_tube_proposals(raw_clips)
        scores = []
        for clip_prop in tube_proposals:
            overlap = 0
            # probability that action is found in the i-th clip
            action_prob = self.compute_actionness(clip_prop[:, -1])
            for j, tube_prop in enumerate(clip_prop):
                overlap += self.compute_overlap(tube_prop[:-1],
                                                clip_prop[j + 1, :-1])
            score = self.compute_score(action_prob, overlap)
            scores.append(score)
        best_prop_idx = int(np.argmax(scores))
        best_seq = tube_proposals[best_prop_idx]
        return best_seq

    @staticmethod
    def compute_overlap(tp1: np.array, tp2: np.array) -> float:
        """
            Compute IoU between last and first frames
            in j-th and j+1-th proposals

        """
        x_a = max(tp1[0], tp2[0])
        y_a = max(tp1[1], tp2[1])
        x_b = min(tp1[2], tp2[2])
        y_b = min(tp1[3], tp2[3])

        # compute the intersection area
        inters_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
        if inters_area == 0:
            return 0
        # compute the area of both boxes
        tp1_area = abs((tp1[2] - tp1[0]) * (tp1[3] - tp1[1]))
        tp2_area = abs((tp2[2] - tp2[0]) * (tp2[3] - tp2[1]))
        iou = inters_area / float(tp1_area + tp2_area - inters_area)

        return iou

    @staticmethod
    def compute_actionness(actionness: np.array) -> torch.float:
        """
        Compute actionness score of paired tubes
        """
        acts = torch.sum(actionness, dim=0)
        return acts

    def compute_score(self, actss: torch.float, overlap: float) -> torch.float:
        """
        Compute tube score, following formula in TCNN paper
        """
        score = 1 / self.n_clips * actss + 1 / (self.n_clips - 1) * overlap
        return score


class ToiPool(nn.Module):
    """
    Tube of Interest pooling layer
    """
    def __init__(self, d: int, h: int, w: int):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool3d((d, h, w))

    def forward(self, tube_props: np.array) -> torch.tensor:
        return self.pool(tube_props)


class BBoxRegressor(nn.Module):
    """
    FasterRCNN's RPN-like subnet to get bounding boxes
    and actionness scores from feature map
    """
    def __init__(self, input_size: tuple, n_boxes: int = 9):
        super().__init__()
        self.n_boxes = n_boxes
        self.c1_1 = nn.Conv1d(512, 36, (1, 1))
        self.c1_2 = nn.Conv1d(512, 9, (1, 1))
        h, w = input_size
        self.fc_bboxes = nn.Linear(h * w * 36, 36)
        self.fc_ascores = nn.Linear(h * w * 9, 9)

    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        x = torch.squeeze(x, dim=2)
        bboxes = self.c1_1(x)
        bboxes = bboxes.view(-1)
        bboxes = self.fc_bboxes(bboxes).reshape(9, 4)
        action_scores = self.c1_2(x)
        action_scores = action_scores.view(-1)
        action_scores = self.fc_ascores(action_scores).reshape(9, 1)
        return bboxes, action_scores


class TPN:
    """
    Generate relevant tube proposals for a clip out of 8 frames
    """
    def __init__(self, in_channels: int, fc6_units: int = 256,
                 fc7_units: int = 512, fc8_units: int = 1024, output: int = 36):
        """Initialize parameters and build model.

        :int    fc6_units : Number of nodes in first hidden layer
        :int    fc7_units : Number of nodes in second hidden layer
        """

        self.in_channels = in_channels
        self.n_anchor = 9  # no. of anchors at each location
        self.toi2 = ToiPool(8, 8, 8)
        self.toi5 = ToiPool(1, 4, 4)
        self.conv11 = nn.Conv1d(144, 8192, 1)
        self.fc6 = nn.Linear(512 * 8 * fc6_units, fc7_units)
        self.fc7 = nn.Linear(fc7_units, fc8_units)
        self.fc8 = nn.Linear(fc8_units, output)

    def update_boxes(self, bboxes: torch.tensor, conv2: torch.tensor):
        """
        Refine bounding boxes proposed by BBoxRegressor 
        :tensor bboxes: bboxes from conv5 (19x25x9x4); 4 -> H, W, x, y
        :tensor conv2: conv2 feature cube (64x8x150x200)
        """

        pdb.set_trace()

        scaled_bboxes = bboxes.detach().clone()
        scaled_bboxes[:, [0, 3]] *= 200 / 19  # update H, y
        scaled_bboxes[:, [1, 2]] *= 150 / 25  # update W, x
        scaled_bboxes = scaled_bboxes.numpy().astype(np.int64)

        # slice tubes from conv2 feature map
        tubes = conv2.data[:, :, :,
                           scaled_bboxes[:, 2] - scaled_bboxes[:, 1] / 2:
                           scaled_bboxes[:, 2] + scaled_bboxes[:, 1] / 2,
                           scaled_bboxes[:, 3] - scaled_bboxes[:, 0] / 2:
                           scaled_bboxes[:, 3] + scaled_bboxes[:, 0] / 2]

        x1 = self.toi2(tubes)
        x1 = torch.norm(x1, p=2)  # Cx8x8x8
        x2 = self.toi5(bboxes)
        x2 = x2.repeat(1, 8, 1, 1, 1)  # Cx8x4x4
        x2 = torch.norm(x2, p=2)
        x = torch.cat((x1, x2), dim=1)  # Cx8x12x12
        x = x.reshape((512, 8, -1))  # CxDx144
        x = self.conv11(x)
        reg = torch.flatten(x)

        reg = self.fc6(reg)
        reg = self.fc7(reg)
        reg = self.fc8(reg)
        return reg


class TCNN(nn.Module):
    """End-to-end action detection model"""

    def __init__(self, input_size: tuple, seed: int, fc8_units: int = 4096):
        """Initialize parameters and build model.
        :tuple    input_size (tuple): (H, W, D) triplet
        :int      seed : Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.input_size = input_size
        self.n_anchor = 9  # no. of anchors at each location
        self.conv1 = nn.Conv3d(3, 64, (3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.conv3b = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)
        self.conv4a = nn.Conv3d(256, 512, (3, 3, 3), padding=1)
        self.conv4b = nn.Conv3d(512, 512, (3, 3, 3), padding=1)
        self.pool4 = nn.MaxPool3d((2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, (3, 3, 3), padding=1)
        self.conv5b = nn.Conv3d(512, 512, (3, 3, 3), padding=1)
        self.bbox_regressor = BBoxRegressor((19, 25))
        self.tpn = TPN(512)
        self.linker = Linker()
        self.reg_layer = nn.Conv3d(fc8_units, self.n_anchor * 4, 1, 1, 0)
        self.cls_layer = nn.Conv3d(fc8_units, self.n_anchor * 2, 1, 1, 0)
        self.toi = ToiPool(8, 9, 9)
        self.fc6 = nn.Linear(648, 512)
        self.drop1 = nn.Dropout(0.25)
        self.fc7 = nn.Linear(512, N_CLASSES)

    def forward(self, x):
        """Build a network that maps anchor boxes to a seq. of frames."""

        x = self.pool1(F.leaky_relu(self.conv1(x)))
        conv2 = F.leaky_relu(self.conv2(x))
        x = self.pool2(conv2)
        x = self.pool3(F.leaky_relu(
            self.conv3b(F.leaky_relu(self.conv3a(x)))))

        x = self.pool4(F.leaky_relu(
            self.conv4b(F.leaky_relu(self.conv4a(x)))))
        x = F.leaky_relu(self.conv5b(F.leaky_relu(self.conv5a(x))))

        bboxes, actionness = self.bbox_regressor(x)
        ref_bboxes = self.tpn.update_boxes(bboxes, conv2.detach().clone())
        tube_props = ref_bboxes.repeat(8, 1, 1)  # for each frame 9 boxes with (x,y,h,w)

        ref_tube_props = self.linker.link_proposals(tube_props)
        x = self.toi(ref_tube_props)
        x = torch.flatten(x)
        x = F.relu(self.fc6(x))
        x = self.drop1(x)
        x = F.relu(self.fc7(x))
        return x
