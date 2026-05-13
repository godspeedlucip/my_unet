"""
各向异性注意力模块，对标 nnUNet network_modules.anisotropic_attention。

抑制有限角度反投影产生的方向性条纹伪影。

三路并行的各向异性 3D 卷积:
  - 轴向 (1,3,3)  ← XY 平面的条纹
  - 矢状 (3,1,3)  ← XZ 平面的条纹
  - 冠状 (3,3,1)  ← YZ 平面的条纹

用 SE 通道注意力自适应融合三路特征 + Sigmoid 门控产生软注意力图。
残差连接确保模块学不到有用信息时退化为恒等变换。
"""

import torch
from torch import nn


class AnisotropicAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()

        # 三路各向异性卷积分支
        self.conv_axial = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3),
                                    padding=(0, 1, 1))
        self.conv_sag = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 3),
                                  padding=(1, 0, 1))
        self.conv_cor = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1),
                                  padding=(1, 1, 0))

        self.norm_axial = nn.InstanceNorm3d(in_channels)
        self.norm_sag = nn.InstanceNorm3d(in_channels)
        self.norm_cor = nn.InstanceNorm3d(in_channels)

        # 融合 & 归一化
        fusion_in = in_channels * 3
        self.fusion = nn.Conv3d(fusion_in, in_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.act = nn.LeakyReLU(inplace=True)

        # SE 通道注意力
        reduced = max(fusion_in // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(fusion_in, reduced, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(reduced, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Sigmoid 门控
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Initialize gate bias so sigmoid ~ 0 -> near-identity at start
        nn.init.constant_(self.gate[0].bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_axial = self.norm_axial(self.conv_axial(x))
        f_sag = self.norm_sag(self.conv_sag(x))
        f_cor = self.norm_cor(self.conv_cor(x))

        concat = torch.cat([f_axial, f_sag, f_cor], dim=1)
        se_weight = self.se(concat)

        fused = self.fusion(concat)
        fused = fused * se_weight
        fused = self.norm(fused)
        fused = self.act(fused)

        attn = self.gate(fused)
        return x * attn + x  # 残差连接
