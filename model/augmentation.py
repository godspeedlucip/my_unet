"""
在线数据增强模块，对标 nnUNet batchgeneratorsv2 transforms。

架构对标 batchgeneratorsv2:
- BasicTransform 基类: get_parameters() → apply() → _apply_to_image / _apply_to_segmentation
- ImageOnlyTransform: 只作用于 image
- RandomTransform: 概率 wrapper, 解耦概率和变换逻辑

所有变换在 torch tensor 上操作，SpatialTransform 利用 GPU grid_sample 加速。
"""

import abc
import numpy as np
import torch
from torch.nn.functional import grid_sample, interpolate, pad, conv3d, conv2d, conv1d
from typing import Tuple, List, Union
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════════════════════════════
# 基类
# ═══════════════════════════════════════════════════════════════════════

class BasicTransform(abc.ABC):
    """
    对标 batchgeneratorsv2.transforms.base.basic_transform.BasicTransform

    约定:
    - data_dict keys: 'image' (C,D,H,W), 'segmentation' (D,H,W) 或 (C,D,H,W)
    - 所有 tensor 都是 torch tensor
    """

    def __call__(self, **data_dict) -> dict:
        params = self.get_parameters(**data_dict)
        return self.apply(data_dict, **params)

    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)
        if data_dict.get('segmentation') is not None:
            data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)
        return data_dict

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return img

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation


class ImageOnlyTransform(BasicTransform):
    """只作用于 image，不碰 segmentation"""

    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)
        return data_dict


# ═══════════════════════════════════════════════════════════════════════
# 工具类
# ═══════════════════════════════════════════════════════════════════════

class RandomTransform(BasicTransform):
    """
    对标 batchgeneratorsv2.transforms.utils.random.RandomTransform
    给任意 transform 加概率 wrapper，解耦概率和变换逻辑。
    """

    def __init__(self, transform: BasicTransform, p: float = 1.0):
        super().__init__()
        self.transform = transform
        self.p = p

    def get_parameters(self, **data_dict) -> dict:
        return {"apply_transform": torch.rand(1).item() < self.p}

    def apply(self, data_dict, **params):
        if params['apply_transform']:
            return self.transform(**data_dict)
        return data_dict


class ComposeTransforms(BasicTransform):
    """对标 batchgeneratorsv2.transforms.utils.compose.ComposeTransforms"""

    def __init__(self, transforms: List[BasicTransform]):
        super().__init__()
        self.transforms = transforms

    def apply(self, data_dict, **params):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict


# ═══════════════════════════════════════════════════════════════════════
# 空间变换
# ═══════════════════════════════════════════════════════════════════════

class MirrorTransform(BasicTransform):
    """
    对标 batchgeneratorsv2.transforms.spatial.mirroring.MirrorTransform
    沿随机轴翻转 (每个轴 50% 概率)
    """

    def __init__(self, allowed_axes: Tuple[int, ...] = (0, 1, 2)):
        super().__init__()
        self.allowed_axes = allowed_axes

    def get_parameters(self, **data_dict) -> dict:
        axes = [i for i in self.allowed_axes if torch.rand(1) < 0.5]
        return {'axes': axes}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['axes']:
            return img
        axes = [a + 1 for a in params['axes']]  # +1 跳过通道维
        return torch.flip(img, axes)

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        if not params['axes']:
            return seg
        # seg 可能没有通道维 (D,H,W) 或有通道维 (C,D,H,W)
        # image 有 C 维，所以 axis 需要 +1；seg 没有 C 维则不用 +1
        offset = 1 if seg.ndim >= 4 else 0
        axes = [a + offset for a in params['axes']]
        return torch.flip(seg, axes)


class SpatialTransform(BasicTransform):
    """
    对标 batchgeneratorsv2.transforms.spatial.spatial.SpatialTransform

    在 GPU 上做旋转 + 缩放 (弹性形变默认关闭)。
    使用 torch.nn.functional.grid_sample 进行可微空间变换。

    Args:
        patch_size: 输出 patch 尺寸 (D, H, W)
        p_rotation: 旋转概率
        rotation_angle_range: 旋转角度范围 (弧度), 3D 默认 (-0.52, 0.52) ≈ ±30°
        p_scaling: 缩放概率
        scaling_range: 缩放因子范围
        p_elastic_deform: 弹性形变概率 (默认 0, 因为较慢)
        elastic_deform_scale: 形变尺度 (% of patch)
        elastic_deform_magnitude: 形变幅度 (像素)
    """

    def __init__(self,
                 patch_size: Tuple[int, ...],
                 p_rotation: float = 0.2,
                 rotation_angle_range: Tuple[float, float] = (-0.52, 0.52),
                 p_scaling: float = 0.2,
                 scaling_range: Tuple[float, float] = (0.7, 1.4),
                 p_elastic_deform: float = 0.0,
                 elastic_deform_scale: Tuple[float, float] = (0, 0.2),
                 elastic_deform_magnitude: Tuple[float, float] = (0, 0.2)):
        super().__init__()
        self.patch_size = patch_size
        self.p_rotation = p_rotation
        self.rotation_angle_range = rotation_angle_range
        self.p_scaling = p_scaling
        self.scaling_range = scaling_range
        self.p_elastic_deform = p_elastic_deform
        self.elastic_deform_scale = elastic_deform_scale
        self.elastic_deform_magnitude = elastic_deform_magnitude

    def get_parameters(self, **data_dict) -> dict:
        dim = 3

        do_rotation = np.random.random() < self.p_rotation
        do_scale = np.random.random() < self.p_scaling
        do_deform = np.random.random() < self.p_elastic_deform

        if do_rotation:
            angles = [np.random.uniform(*self.rotation_angle_range) for _ in range(3)]
        else:
            angles = [0.0] * 3

        if do_scale:
            scales = [np.random.uniform(*self.scaling_range) for _ in range(3)]
        else:
            scales = [1.0] * 3

        if do_scale or do_rotation:
            affine = _create_affine_matrix_3d(angles, scales)
        else:
            affine = None

        if do_deform:
            offsets = _create_elastic_deformation(
                self.patch_size,
                np.random.uniform(*self.elastic_deform_scale),
                np.random.uniform(*self.elastic_deform_magnitude)
            )
        else:
            offsets = None

        return {'affine': affine, 'elastic_offsets': offsets}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self._spatial_transform(img, params, mode='bilinear', padding_mode='zeros')

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        # 给 seg 加通道维以匹配 grid_sample 的输入要求
        if seg.ndim == 3:
            seg = seg.unsqueeze(0)
        was_3d = seg.ndim == 4 and seg.shape[0] == 1
        result = self._spatial_transform(seg.float(), params, mode='nearest', padding_mode='zeros')
        if was_3d:
            result = result.squeeze(0)
        return result.to(seg.dtype)

    def _spatial_transform(self, tensor: torch.Tensor, params, mode='bilinear', padding_mode='zeros'):
        if params['affine'] is None and params['elastic_offsets'] is None:
            return tensor

        spatial_shape = tensor.shape[1:]  # (D, H, W)
        grid = _create_centered_identity_grid(self.patch_size)

        if params['elastic_offsets'] is not None:
            grid = grid + params['elastic_offsets']

        if params['affine'] is not None:
            grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

        # 转换到 grid_sample 坐标系
        for d in range(3):
            grid[..., d] /= (spatial_shape[d] / 2.0) if spatial_shape[d] > 0 else 1.0
        grid = torch.flip(grid, (-1,))  # (D,H,W,3) → (D,H,W,z,y,x)

        return grid_sample(
            tensor[None], grid[None],
            mode=mode, padding_mode=padding_mode, align_corners=False
        )[0]


class Rot90Transform(BasicTransform):
    """对标 batchgeneratorsv2.transforms.spatial.rot90.Rot90Transform"""

    def __init__(self, p=0.5, allowed_axes=(1, 2)):
        """默认只在 H-W 面旋转 (医学影像通常在轴面有最好的分辨率)"""
        super().__init__()
        self.p = p
        self.allowed_axes = allowed_axes

    def get_parameters(self, **data_dict) -> dict:
        if torch.rand(1).item() < self.p:
            from itertools import combinations
            axes_pairs = list(combinations(self.allowed_axes, 2))
            if not axes_pairs:
                return {'axis_pair': None, 'k': 0}
            pair = axes_pairs[np.random.randint(len(axes_pairs))]
            k = np.random.randint(1, 4)  # 1-3 次 90 度
            return {'axis_pair': (pair[0] + 1, pair[1] + 1), 'k': k}  # +1 跳过通道维
        return {'axis_pair': None, 'k': 0}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if params['axis_pair'] is None:
            return img
        return torch.rot90(img, k=params['k'], dims=params['axis_pair'])

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        if params['axis_pair'] is None:
            return seg
        # axis_pair 是 image-space (已 +1 跳过通道维)
        # seg (D,H,W): 没有通道维，需要 -1 → offset=1
        # seg (C,D,H,W): 有通道维，保持 → offset=0
        offset = 1 if seg.ndim < 4 else 0
        seg_axes = (params['axis_pair'][0] - offset, params['axis_pair'][1] - offset)
        return torch.rot90(seg, k=params['k'], dims=seg_axes)


# ═══════════════════════════════════════════════════════════════════════
# 强度变换 (ImageOnlyTransform)
# ═══════════════════════════════════════════════════════════════════════

class GaussianNoiseTransform(ImageOnlyTransform):
    """
    对标 batchgeneratorsv2.transforms.intensity.gaussian_noise.GaussianNoiseTransform
    """

    def __init__(self, noise_variance: Tuple[float, float] = (0, 0.1),
                 p_per_channel: float = 1.0, synchronize_channels: bool = False):
        super().__init__()
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.rand(shape[0]) < self.p_per_channel
        if self.synchronize_channels:
            sigma = np.random.uniform(*self.noise_variance)
            sigmas = [sigma] * int(apply_to_channel.sum().item())
        else:
            sigmas = [np.random.uniform(*self.noise_variance)
                      for _ in range(int(apply_to_channel.sum().item()))]
        return {'apply_to_channel': apply_to_channel, 'sigmas': sigmas}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['apply_to_channel'].any():
            return img
        ch_indices = torch.where(params['apply_to_channel'])[0]
        spatial_shape = (1, *img.shape[1:])
        for idx, ch in enumerate(ch_indices):
            noise = torch.normal(0, params['sigmas'][idx], size=spatial_shape)
            img[ch] += noise[0]
        return img


class GaussianBlurTransform(ImageOnlyTransform):
    """
    对标 batchgeneratorsv2.transforms.noise.gaussian_blur.GaussianBlurTransform

    使用可分离 1D 高斯滤波，避免 3D 卷积的巨大开销。
    """

    def __init__(self, blur_sigma: Tuple[float, float] = (0.5, 1.0),
                 p_per_channel: float = 1.0, synchronize_channels: bool = False):
        super().__init__()
        self.blur_sigma = blur_sigma
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        dims = len(shape) - 1
        apply_to_channel = torch.rand(shape[0]) < self.p_per_channel
        num_ch = int(apply_to_channel.sum().item())
        if self.synchronize_channels:
            s = np.random.uniform(*self.blur_sigma)
            sigmas = [[s] * dims for _ in range(num_ch)]
        else:
            sigmas = [[np.random.uniform(*self.blur_sigma) for _ in range(dims)]
                      for _ in range(num_ch)]
        return {'apply_to_channel': apply_to_channel, 'sigmas': sigmas}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['apply_to_channel'].any():
            return img
        ch_indices = torch.where(params['apply_to_channel'])[0]
        dim = img.ndim - 1
        for idx, ch in enumerate(ch_indices):
            for d in range(dim):
                sigma = params['sigmas'][idx][d]
                if sigma > 0.01:
                    img[ch:ch+1] = _blur_dimension(img[ch:ch+1], sigma, d)
        return img


class MultiplicativeBrightnessTransform(ImageOnlyTransform):
    """
    对标 batchgeneratorsv2.transforms.intensity.brightness.MultiplicativeBrightnessTransform
    """

    def __init__(self, multiplier_range: Tuple[float, float] = (0.75, 1.25),
                 p_per_channel: float = 1.0, synchronize_channels: bool = False):
        super().__init__()
        self.multiplier_range = multiplier_range
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.rand(shape[0]) < self.p_per_channel
        num_ch = int(apply_to_channel.sum().item())
        if self.synchronize_channels:
            m = np.random.uniform(*self.multiplier_range)
            multipliers = [m] * num_ch
        else:
            multipliers = [np.random.uniform(*self.multiplier_range) for _ in range(num_ch)]
        return {'apply_to_channel': apply_to_channel, 'multipliers': multipliers}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['apply_to_channel'].any():
            return img
        ch_indices = torch.where(params['apply_to_channel'])[0]
        for idx, ch in enumerate(ch_indices):
            img[ch] *= params['multipliers'][idx]
        return img


class ContrastTransform(ImageOnlyTransform):
    """
    对标 batchgeneratorsv2.transforms.intensity.contrast.ContrastTransform
    """

    def __init__(self, contrast_range: Tuple[float, float] = (0.75, 1.25),
                 preserve_range: bool = True, p_per_channel: float = 1.0,
                 synchronize_channels: bool = False):
        super().__init__()
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.rand(shape[0]) < self.p_per_channel
        num_ch = int(apply_to_channel.sum().item())
        if self.synchronize_channels:
            c = np.random.uniform(*self.contrast_range)
            multipliers = [c] * num_ch
        else:
            multipliers = [np.random.uniform(*self.contrast_range) for _ in range(num_ch)]
        return {'apply_to_channel': apply_to_channel, 'multipliers': multipliers}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['apply_to_channel'].any():
            return img
        ch_indices = torch.where(params['apply_to_channel'])[0]
        for idx, ch in enumerate(ch_indices):
            mean = img[ch].mean()
            minm, maxm = (img[ch].min(), img[ch].max()) if self.preserve_range else (None, None)
            img[ch] -= mean
            img[ch] *= params['multipliers'][idx]
            img[ch] += mean
            if self.preserve_range:
                img[ch].clamp_(minm, maxm)
        return img


class GammaTransform(ImageOnlyTransform):
    """
    对标 batchgeneratorsv2.transforms.intensity.gamma.GammaTransform

    Gamma 校正: out = ((x - min) / range)^gamma * range + min
    """

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5),
                 p_invert: float = 0.0, p_per_channel: float = 1.0,
                 synchronize_channels: bool = False, p_retain_stats: float = 0.0):
        super().__init__()
        self.gamma_range = gamma_range
        self.p_invert = p_invert
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels
        self.p_retain_stats = p_retain_stats

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.rand(shape[0]) < self.p_per_channel
        num_ch = int(apply_to_channel.sum().item())
        if self.synchronize_channels:
            g = np.random.uniform(*self.gamma_range)
            gammas = [g] * num_ch
        else:
            gammas = [np.random.uniform(*self.gamma_range) for _ in range(num_ch)]
        retain_stats = [np.random.random() < self.p_retain_stats for _ in range(num_ch)]
        invert = [np.random.random() < self.p_invert for _ in range(num_ch)]
        return {'apply_to_channel': apply_to_channel, 'gammas': gammas,
                'retain_stats': retain_stats, 'invert': invert}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['apply_to_channel'].any():
            return img
        ch_indices = torch.where(params['apply_to_channel'])[0]
        for idx, ch in enumerate(ch_indices):
            if params['invert'][idx]:
                img[ch] = -img[ch]

            if params['retain_stats'][idx]:
                mean_before = img[ch].mean()
                std_before = img[ch].std()

            minm = img[ch].min()
            rnge = img[ch].max() - minm
            img[ch] = torch.pow(
                (img[ch] - minm) / max(rnge, 1e-7), params['gammas'][idx]
            ) * rnge + minm

            if params['retain_stats'][idx]:
                mean_after = img[ch].mean()
                std_after = img[ch].std()
                img[ch] = img[ch] - mean_after
                img[ch] = img[ch] * (std_before / max(std_after, 1e-7))
                img[ch] = img[ch] + mean_before

            if params['invert'][idx]:
                img[ch] = -img[ch]
        return img


class SimulateLowResolutionTransform(ImageOnlyTransform):
    """
    对标 batchgeneratorsv2.transforms.spatial.low_resolution.SimulateLowResolutionTransform

    降采样再升采样来模拟低分辨率输入。
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.0),
                 p_per_channel: float = 1.0, synchronize_channels: bool = False,
                 synchronize_axes: bool = False):
        super().__init__()
        self.scale_range = scale_range
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        dims = len(shape) - 1
        apply_to_channel = torch.rand(shape[0]) < self.p_per_channel
        num_ch = int(apply_to_channel.sum().item())
        if self.synchronize_channels:
            if self.synchronize_axes:
                s = np.random.uniform(*self.scale_range)
                scales = [[s] * dims for _ in range(num_ch)]
            else:
                scales = [[np.random.uniform(*self.scale_range) for _ in range(dims)]
                          for _ in range(num_ch)]
        else:
            if self.synchronize_axes:
                scales = [[np.random.uniform(*self.scale_range)] * dims
                          for _ in range(num_ch)]
            else:
                scales = [[np.random.uniform(*self.scale_range) for _ in range(dims)]
                          for _ in range(num_ch)]
        return {'apply_to_channel': apply_to_channel, 'scales': scales}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params['apply_to_channel'].any():
            return img
        ch_indices = torch.where(params['apply_to_channel'])[0]
        orig_shape = img.shape[1:]
        upmode = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}[img.ndim - 1]
        for idx, ch in enumerate(ch_indices):
            new_shape = [max(2, round(s * float(orig_shape[d])))
                         for d, s in enumerate(params['scales'][idx])]
            downsampled = interpolate(img[ch][None, None], size=new_shape, mode='nearest-exact')
            img[ch] = interpolate(downsampled, size=orig_shape, mode=upmode)[0, 0]
        return img


# ═══════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _create_affine_matrix_3d(angles, scales):
    """创建 3D 旋转 + 缩放仿射矩阵。对标 nnUNet 的 create_affine_matrix_3d"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    S = np.diag(scales)
    return Rz @ Ry @ Rx @ S


def _create_centered_identity_grid(size: Tuple[int, ...]) -> torch.Tensor:
    """创建居中于 (0,0,0) 的 identity grid。对标 nnUNet"""
    space = [torch.linspace((1 - s) / 2, (s - 1) / 2, s) for s in size]
    grid = torch.meshgrid(space, indexing="ij")
    grid = torch.stack(grid, -1)  # (D, H, W, 3)
    return grid


def _create_elastic_deformation(patch_size, scale, magnitude):
    """创建弹性形变偏移场。使用 scipy gaussian_filter 做平滑。"""
    dim = len(patch_size)
    offsets = np.random.randn(dim, *patch_size).astype(np.float32)
    sigma = scale * np.array(patch_size)
    for d in range(dim):
        offsets[d] = gaussian_filter(offsets[d], sigma[d])
        mx = np.max(np.abs(offsets[d]))
        if mx > 1e-8:
            offsets[d] /= (mx / max(magnitude, 1e-8))
    offsets = torch.from_numpy(offsets)
    spatial_dims = tuple(range(1, dim + 1))
    offsets = offsets.permute(*spatial_dims, 0)  # (D, H, W, 3)
    return offsets


def _build_kernel(sigma: float, truncate: float = 6.0) -> torch.Tensor:
    """构建 1D 高斯核。对标 nnUNet _build_kernel"""
    ksize = round(sigma * truncate + 0.5)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    half = (ksize - 1) * 0.5
    x = torch.linspace(-half, half, steps=ksize)
    pdf = torch.exp(-0.5 * (x / sigma) ** 2)
    return pdf / pdf.sum()


def _blur_dimension(img: torch.Tensor, sigma: float, dim_to_blur: int) -> torch.Tensor:
    """
    对标 nnUNet blur_dimension:
    沿指定维度做可分离 1D 高斯模糊。
    """
    kernel = _build_kernel(sigma).to(img.device)
    ksize = kernel.shape[0]
    spatial_dims = img.ndim - 1

    if spatial_dims == 3:
        if dim_to_blur == 0:
            kernel = kernel[None, None, :, None, None]
            padding = [0, 0, 0, 0, ksize // 2, ksize // 2]
        elif dim_to_blur == 1:
            kernel = kernel[None, None, None, :, None]
            padding = [0, 0, ksize // 2, ksize // 2, 0, 0]
        else:
            kernel = kernel[None, None, None, None, :]
            padding = [ksize // 2, ksize // 2, 0, 0, 0, 0]
        conv_op = conv3d
    elif spatial_dims == 2:
        if dim_to_blur == 0:
            kernel = kernel[None, None, :, None]
            padding = [0, 0, ksize // 2, ksize // 2]
        else:
            kernel = kernel[None, None, None, :]
            padding = [ksize // 2, ksize // 2, 0, 0]
        conv_op = conv2d
    else:
        kernel = kernel[None, None, :]
        padding = [ksize // 2, ksize // 2]
        conv_op = conv1d

    img_padded = pad(img, padding, mode="reflect")
    n_channels = img_padded.shape[0]
    kernel_expanded = kernel.expand(n_channels, *([-1] * (kernel.ndim - 1)))
    return conv_op(img_padded[None], kernel_expanded, groups=n_channels)[0]


# ═══════════════════════════════════════════════════════════════════════
# 获取默认增强流水线 (对标 nnUNet nnUNetTrainer.get_training_transforms)
# ═══════════════════════════════════════════════════════════════════════

def get_default_augmentation(stage='train', patch_size=(128, 128, 128)):
    """
    获取默认增强流水线。

    对标 nnUNetV2 nnUNetTrainer 的 get_training_transforms:
    - SpatialTransform (rotation 20%, scaling 20%)
    - GaussianNoiseTransform (10%)
    - GaussianBlurTransform (20%)
    - MultiplicativeBrightnessTransform (15%)
    - ContrastTransform (15%)
    - SimulateLowResolutionTransform (25%)
    - GammaTransform inverted (10%)
    - GammaTransform non-inverted (30%)
    - MirrorTransform (all axes, 50%)

    Args:
        stage: 'train' 或 'val'
        patch_size: 3D patch 尺寸
    """
    if stage == 'train':
        return ComposeTransforms([
            SpatialTransform(
                patch_size=patch_size,
                p_rotation=0.2,
                p_scaling=0.2,
            ),
            RandomTransform(GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1.0,
                synchronize_channels=True,
            ), p=0.1),
            RandomTransform(GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                p_per_channel=1.0,
                synchronize_channels=False,
            ), p=0.2),
            RandomTransform(MultiplicativeBrightnessTransform(
                multiplier_range=(0.75, 1.25),
                p_per_channel=1.0,
            ), p=0.15),
            RandomTransform(ContrastTransform(
                contrast_range=(0.75, 1.25),
                preserve_range=True,
                p_per_channel=1.0,
            ), p=0.15),
            RandomTransform(SimulateLowResolutionTransform(
                scale_range=(0.5, 1.0),
                p_per_channel=1.0,
                synchronize_channels=True,
                synchronize_axes=True,
            ), p=0.25),
            RandomTransform(GammaTransform(
                gamma_range=(0.7, 1.5),
                p_invert=1.0,
                p_per_channel=1.0,
            ), p=0.1),
            RandomTransform(GammaTransform(
                gamma_range=(0.7, 1.5),
                p_invert=0.0,
                p_per_channel=1.0,
            ), p=0.3),
            MirrorTransform(allowed_axes=(0, 1, 2)),
        ])
    else:
        return ComposeTransforms([])
