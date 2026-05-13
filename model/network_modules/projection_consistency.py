"""
投影一致性损失，对标 nnUNet network_modules.projection_consistency。

利用原始投影图做自监督约束：对预测的 3D 分割 mask 做可微前向投影，
与 ground truth 的投影做 MSE。

核心组件:
- build_rotation_matrix_3d()        — 绕 Z 轴的 3x3 旋转矩阵
- differentiable_forward_projection() — 通过 F.affine_grid + F.grid_sample
  旋转体积后沿 Z 轴积分，全程可微
- ProjectionConsistencyLoss         — MSE(预测投影, 真值投影)
"""

import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_rotation_matrix_3d(angle_rad: float, device: torch.device,
                             dtype: torch.dtype) -> torch.Tensor:
    """构建绕 Z 轴的 3x3 旋转矩阵"""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    R = torch.tensor([
        [cos_a, sin_a, 0],
        [-sin_a, cos_a, 0],
        [0, 0, 1],
    ], device=device, dtype=dtype)
    return R


def _build_affine_cache(angles_deg: List[float], device: torch.device,
                        dtype: torch.dtype) -> Dict[float, torch.Tensor]:
    """预计算各投影角度的 (1, 3, 4) 仿射矩阵"""
    cache = {}
    for angle_deg in angles_deg:
        theta = math.radians(angle_deg)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        aff = torch.zeros(1, 3, 4, device=device, dtype=dtype)
        aff[0, 0, 0] = cos_a
        aff[0, 0, 1] = sin_a
        aff[0, 1, 0] = -sin_a
        aff[0, 1, 1] = cos_a
        aff[0, 2, 2] = 1.0
        cache[angle_deg] = aff
    return cache


def differentiable_forward_projection(
    volume: torch.Tensor,
    angles_deg: Union[List[float], np.ndarray],
    align_corners: bool = False,
    affine_cache: Optional[Dict[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    可微平行束前向投影：旋转 + 沿 Z 轴积分 (平均)。

    Args:
        volume: (B, C, D, H, W) float tensor
        angles_deg: 投影角度列表（度）
        align_corners: 传给 grid_sample
        affine_cache: 可选预计算的 angle_deg → (1, 3, 4) affine 字典

    Returns:
        (B, N_views, C, H, W) — 各视角的 2D 投影
    """
    B, C, D, H, W = volume.shape
    N = len(angles_deg)
    device = volume.device
    dtype = volume.dtype

    if affine_cache is None:
        affine_cache = _build_affine_cache(list(angles_deg), device, dtype)
    else:
        sample_aff = next(iter(affine_cache.values()))
        if sample_aff.device != device or sample_aff.dtype != dtype:
            affine_cache = _build_affine_cache(list(angles_deg), device, dtype)

    size = torch.Size([B, C, D, H, W])
    projections = []

    for angle_deg in angles_deg:
        theta_rad = math.radians(angle_deg)

        # 0° 投影退化：直接沿 Z 平均
        if abs(theta_rad) < 1e-8:
            proj = volume.mean(dim=2)
            projections.append(proj)
            continue

        affine = affine_cache[angle_deg].expand(B, -1, -1)
        grid = F.affine_grid(affine, size, align_corners=align_corners)
        rotated = F.grid_sample(volume, grid, mode='bilinear',
                                align_corners=align_corners, padding_mode='zeros')

        proj = rotated.mean(dim=2)  # 沿 Z 平均 → (B, C, H, W)
        projections.append(proj)

    return torch.stack(projections, dim=1)  # (B, N, C, H, W)


class ProjectionConsistencyLoss(nn.Module):
    """
    预测投影与真值投影之间的 MSE。

    Args:
        angles_deg: 投影角度列表（度）
        align_corners: 传给 grid_sample
    """

    def __init__(self, angles_deg: Union[List[float], np.ndarray],
                 align_corners: bool = False):
        super().__init__()
        self.angles_deg = list(angles_deg)
        self.align_corners = align_corners
        self._affine_cache: Optional[Dict[float, torch.Tensor]] = None

    def _get_affine_cache(self, device: torch.device,
                          dtype: torch.dtype) -> Dict[float, torch.Tensor]:
        if self._affine_cache is None:
            self._affine_cache = _build_affine_cache(
                self.angles_deg, device, dtype)
        else:
            sample = next(iter(self._affine_cache.values()))
            if sample.device != device or sample.dtype != dtype:
                self._affine_cache = _build_affine_cache(
                    self.angles_deg, device, dtype)
        return self._affine_cache

    def forward(self, pred_logits: torch.Tensor,
                target_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, C, D, H, W) — softmax 后的概率
            target_onehot: (B, C, D, H, W) — one-hot GT

        Returns:
            标量 MSE loss
        """
        cache = self._get_affine_cache(pred_logits.device, pred_logits.dtype)

        pred_proj = differentiable_forward_projection(
            pred_logits, self.angles_deg, self.align_corners,
            affine_cache=cache)
        target_proj = differentiable_forward_projection(
            target_onehot.float(), self.angles_deg, self.align_corners,
            affine_cache=cache)

        return F.mse_loss(pred_proj, target_proj)


class CombinedSegProjLoss(nn.Module):
    """
    将分割损失与投影一致性损失组合为单一 nn.Module。

    兼容 ./unet 现有训练循环: forward(output, target) → scalar loss。

    Args:
        seg_loss: 监督分割损失 (e.g. CombinedLoss, BCELoss)
        proj_loss: ProjectionConsistencyLoss 实例
        proj_weight: 投影一致性项的权重
        use_softmax: True 对 logits 做 softmax 后投影, False 直接投影 logits
    """

    def __init__(self, seg_loss: nn.Module, proj_loss: ProjectionConsistencyLoss,
                 proj_weight: float = 0.1, use_softmax: bool = True):
        super().__init__()
        self.seg_loss = seg_loss
        self.proj_loss = proj_loss
        self.proj_weight = proj_weight
        self.use_softmax = use_softmax

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_seg = self.seg_loss(output, target)

        if self.use_softmax:
            pred_vol = torch.softmax(output, dim=1)
        else:
            pred_vol = output

        # target 可能是 (B,1,D,H,W) binary mask — 需要转为 one-hot 用于投影
        if target.shape[1] == 1 and pred_vol.shape[1] > 1:
            target_oh = torch.cat([1.0 - target, target], dim=1)
        else:
            target_oh = target.float()

        l_proj = self.proj_loss(pred_vol, target_oh)
        return l_seg + self.proj_weight * l_proj
