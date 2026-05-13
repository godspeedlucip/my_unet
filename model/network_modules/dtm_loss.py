"""
DTM (Distance Transform Map) 损失，对标 nnUNet training/loss/dtm_losses.py。

利用预计算的距离变换图对管状结构分割进行几何约束。
DTM 值表示每个体素到最近边界的距离，在前景管腔中心最小（最负），
在边界处最大（接近 0）。通过最小化 mean(dtm * pred) 推动预测概率
在管腔中心最高、边界处最低。

核心组件:
- get_dtm              — 计算距离变换图 (scipy EDT, 无 ANTsPy 依赖)
- GenSurfLoss          — 广义表面损失: mean(dtm * softmax(pred)[foreground])
- LinearSchedule       — 线性衰减调度器, 用于 alpha 权重
- CombinedDTMLoss      — DC_CE_loss + alpha * GSL_loss (epoch >= 250 时 alpha=0.5)
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_dtm(mask: np.ndarray, voxel_spacing: tuple = (1.0, 1.0, 1.0),
            label_list: list = None) -> np.ndarray:
    """
    计算二值 mask 的距离变换图 (Distance Transform Map)。

    使用 scipy.ndimage.distance_transform_edt (欧几里得距离变换)，
    无需 ANTsPy 依赖。

    对每个标签分别计算:
      - 前景: 正值 EDT (内部→边界距离为正)
      - 背景: 负值 EDT (外部→边界距离为负, 取反)

    Args:
        mask: (D, H, W) 整数类别 mask, 0=背景, 1=前景
        voxel_spacing: (dz, dy, dx) 体素间距
        label_list: 要计算 DTM 的标签列表 (默认 [0, 1])

    Returns:
        (D, H, W, num_labels) float32 DTM
    """
    from scipy.ndimage import distance_transform_edt

    if label_list is None:
        label_list = [0, 1]

    dtm = np.zeros((*mask.shape, len(label_list)), dtype=np.float32)

    for j, label in enumerate(label_list):
        binary = (mask == label).astype(np.uint8)
        if np.sum(binary) == 0:
            # 该标签不存在，填充最大距离
            d = np.sqrt(sum(s**2 * mask.shape[i]**2
                          for i, s in enumerate(voxel_spacing)))
            dtm[..., j] = d
            continue

        # 正距离: 体素到最近边界的距离
        pos_dtm = distance_transform_edt(binary, sampling=voxel_spacing)
        # 负距离: 外部体素到边界的距离
        neg_dtm = distance_transform_edt(1 - binary, sampling=voxel_spacing)

        # 最终 DTM: 前景=正, 背景=负
        dtm[..., j] = np.where(binary, pos_dtm, -neg_dtm)

    return dtm


def calculate_class_percentage(mask: np.ndarray) -> dict:
    """计算各类别体素占比。"""
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    return {int(v): float(c) / total * 100 for v, c in zip(unique, counts)}


class GenSurfLoss(nn.Module):
    """
    简化的广义表面损失: 对前景类计算 mean(dtm * pred_prob)。

    dtm 值在管腔中心最小 (最负), 边界处最大。
    因此 mean(dtm * pred) 应尽可能小 (负)。

    Args:
        ignored_label: 忽略的标签值 (默认 -1)
    """

    def __init__(self, ignored_label: int = -1):
        super().__init__()
        self.ignored_label = ignored_label

    def forward(self, y_pred: torch.Tensor, dtm: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            y_pred: (B, C, D, H, W) — 网络输出 logits
            dtm: (B, 1, D, H, W) — 前景类距离变换图
            valid_mask: (B, 1, D, H, W) — 有效区域 mask (可选)

        Returns:
            标量损失
        """
        y_prob = torch.softmax(y_pred, dim=1)

        # 取前景通道 (class=1)
        if y_prob.shape[1] > 1:
            fg_prob = y_prob[:, 1]
        else:
            fg_prob = y_prob[:, 0]

        if dtm.shape[1] > 1:
            dtm = dtm[:, :1]

        if valid_mask is not None:
            fg_prob = fg_prob * valid_mask[:, 0]
            dtm = dtm * valid_mask

        return torch.mean(dtm * fg_prob)


class LinearSchedule:
    """
    线性衰减调度器: epoch ≤ init_pause 时返回 1.0,
    之后线性衰减至 num_epochs 时返回 0.0。

    Args:
        num_epochs: 总训练轮数
        init_pause: 前 init_pause 轮保持不变
    """

    def __init__(self, num_epochs: int, init_pause: int = 0):
        if num_epochs <= init_pause:
            raise ValueError("num_epochs must be > init_pause")
        self.num_epochs = num_epochs - 1
        self.init_pause = init_pause

    def __call__(self, epoch: int) -> float:
        if epoch > self.num_epochs:
            raise ValueError(f"epoch {epoch} > num_epochs {self.num_epochs}")
        if epoch > self.init_pause:
            return min(1.0, max(0.0,
                1.0 - float(epoch - self.init_pause) / (self.num_epochs - self.init_pause)))
        return 1.0


class CombinedDTMLoss(nn.Module):
    """
    Loss = seg_loss + alpha * GenSurfLoss。

    alpha 在 epoch < 250 时为 0 (warmup), epoch >= 250 时固定为 0.5。

    使用方式:
        loss_fn = CombinedDTMLoss(seg_loss=your_dice_ce_loss)
        # 训练循环中:
        total, dc_ce, gsl, alpha = loss_fn(output, target, dtm, valid_mask, epoch)

    Args:
        seg_loss: 监督分割损失 (e.g. DC+CE)
        gsl_loss: GenSurfLoss 实例 (默认自动创建)
        gsl_weight: GenSurfLoss 达到稳定后的权重 (默认 0.5)
        gsl_warmup_epochs: 多少轮后开始加入 GenSurfLoss (默认 250)
    """

    def __init__(self, seg_loss: nn.Module, gsl_loss: nn.Module = None,
                 gsl_weight: float = 0.5, gsl_warmup_epochs: int = 250):
        super().__init__()
        self.seg_loss = seg_loss
        self.gsl_loss = gsl_loss or GenSurfLoss()
        self.gsl_weight = gsl_weight
        self.gsl_warmup_epochs = gsl_warmup_epochs

    def forward(self, net_output: torch.Tensor, target: torch.Tensor,
                dtm: torch.Tensor, valid_mask: torch.Tensor = None,
                epoch: int = 0):
        """
        Args:
            net_output: (B, C, D, H, W) logits
            target: (B, 1, D, H, W) ground truth
            dtm: (B, 1, D, H, W) distance transform map
            valid_mask: (B, 1, D, H, W) optional valid region mask
            epoch: 当前 epoch (控制 alpha)

        Returns:
            (total_loss, seg_loss, gsl_loss, alpha)
        """
        alpha = self.gsl_weight if epoch >= self.gsl_warmup_epochs else 0.0

        loss_seg = self.seg_loss(net_output, target)
        loss_gsl = self.gsl_loss(net_output, dtm, valid_mask)

        total = loss_seg + alpha * loss_gsl
        return total, loss_seg, loss_gsl, alpha
