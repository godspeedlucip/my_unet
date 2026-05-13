"""
包装网络，对标 nnUNet network_modules.aniso_stent_unet.AnisoStentUNet。

在前端加 AnisotropicAttention（抑制条纹伪影），
在后端加 TubularEnhancement（精炼管状结构）。

兼容 deep supervision：当 backbone 返回列表时，
仅对最高分辨率的最终输出应用 TubularEnhancement，辅助输出原样通过。
"""

from typing import Union, List

import torch
from torch import nn

from .anisotropic_attention import AnisotropicAttention
from .tubular_enhancement import TubularEnhancement


class AnisoStentUNet(nn.Module):
    """
    包装标准 U-Net backbone:

    1. AnisotropicAttention (前置) — 抑制方向性条纹伪影
    2. TubularEnhancement (后置) — 恢复细管状结构

    两个模块都用残差连接，不损害 backbone 已有能力。
    """

    def __init__(self, backbone_unet: nn.Module, in_channels: int, num_classes: int):
        super().__init__()
        self.aniso_attn = AnisotropicAttention(in_channels)
        self.backbone = backbone_unet
        self.tub_enh = TubularEnhancement(num_classes)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.aniso_attn(x)
        out = self.backbone(x)

        if isinstance(out, list):
            # Deep supervision: 只精炼最高分辨率的输出
            out[0] = self.tub_enh(out[0])
            return out

        return self.tub_enh(out)


class _PartialAnisoUNet(nn.Module):
    """按需组合 AnisotropicAttention 和/或 TubularEnhancement 的轻量包装器。

    与 AnisoStentUNet 不同，这个包装器允许单独使用:
      - --aniso_attn   → 仅前置各向异性注意力
      - --tub_enh      → 仅后置管状结构增强
      - 两者同时使用   → 等效于 AnisoStentUNet

    所有模块都有残差连接，学不到有用信息时退化为恒等变换。
    """

    def __init__(self, backbone, in_channels, num_classes,
                 aniso_attn=True, tub_enh=True):
        super().__init__()
        self.backbone = backbone
        self.aniso_attn = AnisotropicAttention(in_channels) if aniso_attn else None
        self.tub_enh = TubularEnhancement(num_classes) if tub_enh else None

    def forward(self, x):
        if self.aniso_attn is not None:
            x = self.aniso_attn(x)
        out = self.backbone(x)
        if self.tub_enh is not None:
            if isinstance(out, list):
                out[0] = self.tub_enh(out[0])
                return out
            return self.tub_enh(out)
        return out
