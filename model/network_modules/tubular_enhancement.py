"""
管状结构增强模块，对标 nnUNet network_modules.tubular_enhancement。

多尺度精炼细薄管状结构（支架、血管等），防止 U-Net 下采样过程中丢失。

4 路并行的膨胀 3D 卷积 (dil=1,2,3,5) 覆盖不同粗细的管状结构，
通过 1x1x1 融合 + 残差连接，学不到有用信息时退化为恒等变换。

输入/输出形状: (B, num_classes, D, H, W)
"""

import torch
from torch import nn
from torch.nn import functional as F


class TubularEnhancement(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        hidden = max(num_classes * 2, 8)

        self.dil1 = nn.Conv3d(num_classes, hidden, kernel_size=3,
                              dilation=1, padding=1)
        self.dil2 = nn.Conv3d(num_classes, hidden, kernel_size=3,
                              dilation=2, padding=2)
        self.dil3 = nn.Conv3d(num_classes, hidden, kernel_size=3,
                              dilation=3, padding=3)
        self.dil5 = nn.Conv3d(num_classes, hidden, kernel_size=3,
                              dilation=5, padding=5)

        self.norm_dil1 = nn.InstanceNorm3d(hidden)
        self.norm_dil2 = nn.InstanceNorm3d(hidden)
        self.norm_dil3 = nn.InstanceNorm3d(hidden)
        self.norm_dil5 = nn.InstanceNorm3d(hidden)

        self.fusion = nn.Sequential(
            nn.Conv3d(hidden * 4, num_classes, kernel_size=1),
            nn.InstanceNorm3d(num_classes),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(num_classes, num_classes, kernel_size=1),
        )

        # Near-zero init so fusion starts near identity but gradients can flow
        nn.init.normal_(self.fusion[3].weight, std=1e-4)
        nn.init.zeros_(self.fusion[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = F.leaky_relu(self.norm_dil1(self.dil1(x)), inplace=True)
        d2 = F.leaky_relu(self.norm_dil2(self.dil2(x)), inplace=True)
        d3 = F.leaky_relu(self.norm_dil3(self.dil3(x)), inplace=True)
        d5 = F.leaky_relu(self.norm_dil5(self.dil5(x)), inplace=True)

        out = self.fusion(torch.cat([d1, d2, d3, d5], dim=1))
        return out + x  # 残差连接
