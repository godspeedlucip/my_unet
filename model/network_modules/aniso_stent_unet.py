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
        self.deep_supervision = getattr(backbone_unet, 'deep_supervision', False)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.aniso_attn(x)
        out = self.backbone(x)

        if isinstance(out, list):
            # Deep supervision: 只精炼最高分辨率的输出
            out[0] = self.tub_enh(out[0])
            return out

        return self.tub_enh(out)
