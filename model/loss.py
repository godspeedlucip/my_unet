import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: (B, 1, D, H, W) after sigmoid
        # targets: (B, 1, D, H, W)
        preds = torch.sigmoid(preds)
        preds = preds.contiguous().view(preds.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / \
               (preds.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1 - dice.mean()


class LaplacianSmoothingLoss(nn.Module):
    def __init__(self):
        super(LaplacianSmoothingLoss, self).__init__()
        # 3D Laplacian kernel
        kernel_3d = torch.tensor([
            [[0, 0, 0],
             [0, -1, 0],
             [0, 0, 0]],

            [[0, -1, 0],
             [-1, 6, -1],
             [0, -1, 0]],

            [[0, 0, 0],
             [0, -1, 0],
             [0, 0, 0]]
        ], dtype=torch.float32)

        self.register_buffer("kernel", kernel_3d.unsqueeze(0).unsqueeze(0))  # (1,1,3,3,3)

    def forward(self, preds):
        preds = torch.sigmoid(preds)
        # print('loss: ',preds.shape,self.kernel.shape)
        lap = F.conv3d(preds, self.kernel, padding=1)
        return torch.mean(lap ** 2)


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, lap_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.dice = SoftDiceLoss()
        self.lap = LaplacianSmoothingLoss()
        self.dice_weight = dice_weight
        self.lap_weight = lap_weight

    def forward(self, preds, targets):
        loss_dice = self.dice(preds, targets)
        loss_lap = self.lap(preds)
        return self.dice_weight * loss_dice + self.lap_weight * loss_lap
