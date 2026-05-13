"""
预处理脚本：将 .nii 数据一次性裁剪、归一化并保存为 .npy + .pkl。

对标 nnUNet 的 DefaultPreprocessor，关键优化：
1. crop_to_nonzero — 裁剪掉大量背景空气区域
2. 前景归一化 — 只在前景区域计算 mean/std
3. sample_foreground_locations — 预计算前景体素坐标，训练时 O(1) 采样
4. 多进程并行处理
5. 输出 .npy 格式，训练时用 mmap_mode='r' 零拷贝读取

用法：
    python preprocess_v2.py --data_dir /path/to/images --gt_dir /path/to/labels --output_dir /path/to/preprocessed
"""

import os
import sys
import argparse
import pickle
import numpy as np
import nibabel as nib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.ndimage import binary_fill_holes


def crop_to_nonzero(data, seg=None):
    """
    找到所有通道中非零区域的紧致边界框并裁剪。
    对标 nnUNet 的 crop_to_nonzero。

    Args:
        data: (D, H, W) 或 (C, D, H, W) 的 numpy 数组
        seg: (D, H, W) 的 numpy 数组，可选

    Returns:
        cropped_data, cropped_seg, bbox
        bbox = [[d_min, d_max], [h_min, h_max], [w_min, w_max]]
    """
    if data.ndim == 3:
        # 对前景区域做 hole-filling，保留被组织包围的空腔
        mask = data != 0
        # 在每个 2D 切片上填充孔洞
        for z in range(mask.shape[0]):
            mask[z] = binary_fill_holes(mask[z])
    else:
        # 多通道：任一通道非零即为前景
        mask = np.any(data != 0, axis=0)
        for z in range(mask.shape[0]):
            mask[z] = binary_fill_holes(mask[z])

    # 找到每个轴的非零区间
    nz = np.argwhere(mask)
    if len(nz) == 0:
        # 全零数据，返回原始尺寸
        bbox = [[0, data.shape[-3]], [0, data.shape[-2]], [0, data.shape[-1]]]
    else:
        bbox = []
        for axis in range(3):
            vals = nz[:, axis]
            bbox.append([int(vals.min()), int(vals.max()) + 1])

    # 裁剪
    if data.ndim == 3:
        cropped_data = data[bbox[0][0]:bbox[0][1],
                            bbox[1][0]:bbox[1][1],
                            bbox[2][0]:bbox[2][1]]
    else:
        cropped_data = data[:,
                            bbox[0][0]:bbox[0][1],
                            bbox[1][0]:bbox[1][1],
                            bbox[2][0]:bbox[2][1]]

    cropped_seg = None
    if seg is not None:
        cropped_seg = seg[bbox[0][0]:bbox[0][1],
                          bbox[1][0]:bbox[1][1],
                          bbox[2][0]:bbox[2][1]]
        # 裁剪区域外的体素标记为 -1（忽略标签）
        seg_old = cropped_seg.copy()
        cropped_seg[~mask[bbox[0][0]:bbox[0][1],
                          bbox[1][0]:bbox[1][1],
                          bbox[2][0]:bbox[2][1]]] = -1

    return cropped_data, cropped_seg, bbox


def sample_foreground_locations(seg, max_per_class=10000):
    """
    预计算每个前景类别的体素坐标，训练时 O(1) 采样前景位置。
    对标 nnUNet 的 _sample_foreground_locations。

    Args:
        seg: (D, H, W) int 数组，0 为背景，>0 为不同类别，-1 为忽略区域
        max_per_class: 每类最多保留的坐标数

    Returns:
        class_locations: {class_id: [(d, h, w), ...], ...}
    """
    class_locations = {}
    present_classes = np.unique(seg)
    present_classes = present_classes[present_classes > 0]  # 排除背景和忽略区域

    for c in present_classes:
        # 找到所有属于该类别的体素坐标
        locs = np.argwhere(seg == int(c))  # (N, 3) — (d, h, w)
        if len(locs) == 0:
            continue
        if len(locs) > max_per_class:
            # 均匀子采样
            indices = np.linspace(0, len(locs) - 1, max_per_class, dtype=int)
            locs = locs[indices]
        class_locations[int(c)] = locs

    return class_locations


def preprocess_case(args):
    """
    处理单个样本的完整流水线。

    Args:
        args: (data_path, seg_path, output_dir)

    Returns:
        case_id, success, error_msg
    """
    data_path, seg_path, output_dir = args

    case_id = os.path.basename(data_path)
    for ext in ('.nii.gz', '.nii'):
        if case_id.endswith(ext):
            case_id = case_id[:-len(ext)]
            break
    # 去掉 nnUNet _0000 模态后缀
    if case_id.endswith('_0000'):
        case_id = case_id[:-5]

    try:
        # 1. 加载数据
        data_obj = nib.load(data_path)
        data = data_obj.get_fdata(dtype=np.float32)  # (D, H, W)
        seg_obj = nib.load(seg_path)
        seg = seg_obj.get_fdata(dtype=np.float32)

        shape_before_cropping = data.shape

        # 2. 裁剪非零区域
        data, seg, bbox = crop_to_nonzero(data, seg)

        # 3. 在前景区域做 Z-Score 归一化
        fg_mask = data != 0
        if fg_mask.sum() > 0:
            mean = data[fg_mask].mean()
            std = data[fg_mask].std()
            if std > 1e-8:
                data = (data - mean) / std
            else:
                data = data - mean
        else:
            # 全零数据
            mean = 0.0
            std = 1.0

        # 4. 预计算前景位置
        class_locations = sample_foreground_locations(seg)

        # 5. 保存
        data_path_out = os.path.join(output_dir, f"{case_id}_data.npy")
        seg_path_out = os.path.join(output_dir, f"{case_id}_seg.npy")
        props_path_out = os.path.join(output_dir, f"{case_id}.pkl")

        np.save(data_path_out, data.astype(np.float32))
        np.save(seg_path_out, seg.astype(np.int16))

        properties = {
            'shape_before_cropping': shape_before_cropping,
            'bbox_used_for_cropping': bbox,
            'class_locations': class_locations,
            'mean': float(mean),
            'std': float(std),
        }
        with open(props_path_out, 'wb') as f:
            pickle.dump(properties, f)

        return case_id, True, None

    except Exception as e:
        return case_id, False, str(e)


def preprocess_dataset(data_dir, gt_dir, output_dir, num_workers=None):
    """
    预处理整个数据集。

    Args:
        data_dir: 原始图像 (.nii) 目录
        gt_dir: 标注 (.nii) 目录
        output_dir: 预处理输出目录
        num_workers: 并行进程数，默认为 CPU 核心数
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有文件，自动处理 nnUNet _0000 后缀
    data_files_raw = sorted([f for f in os.listdir(data_dir) if f.endswith(('.nii', '.nii.gz'))])
    gt_files_raw = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.nii', '.nii.gz'))])

    # nnUNet 格式：图像为 {case}_0000.nii.gz，标签为 {case}.nii.gz
    # 去掉 _0000 后缀建立映射
    def strip_nnunet_suffix(fname):
        base = fname
        for ext in ('.nii.gz', '.nii'):
            if base.endswith(ext):
                base = base[:-len(ext)]
                break
        if base.endswith('_0000'):
            base = base[:-5]
        return base + ('.nii.gz' if fname.endswith('.nii.gz') else '.nii')

    data_map = {strip_nnunet_suffix(f): f for f in data_files_raw}
    gt_map = {strip_nnunet_suffix(f): f for f in gt_files_raw}

    common = sorted(set(data_map.keys()) & set(gt_map.keys()))
    assert len(common) > 0, "data_dir 和 gt_dir 中找不到匹配的文件名对"
    if len(data_map) != len(common) or len(gt_map) != len(common):
        print(f"注意: data_dir 中有 {len(data_map)} 个文件, gt_dir 中有 {len(gt_map)} 个文件, "
              f"匹配到 {len(common)} 对")

    print(f"预处理 {len(common)} 个样本，使用 {num_workers} 个进程...")

    tasks = []
    for case_id in common:
        tasks.append((
            os.path.join(data_dir, data_map[case_id]),
            os.path.join(gt_dir, gt_map[case_id]),
            output_dir
        ))

    success_count = 0
    fail_list = []

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(preprocess_case, tasks),
            total=len(tasks),
            desc="预处理中"
        ))

    for case_id, ok, err in results:
        if ok:
            success_count += 1
        else:
            fail_list.append((case_id, err))

    print(f"\n完成: {success_count}/{len(tasks)} 成功")
    if fail_list:
        print(f"失败 {len(fail_list)} 个:")
        for case_id, err in fail_list:
            print(f"  - {case_id}: {err}")

    return success_count, fail_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理原始 .nii 数据为 .npy 格式")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="原始图像 (.nii) 所在目录")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="标注 (.nii) 所在目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="预处理后的输出目录")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="并行进程数 (默认: CPU 核心数)")
    args = parser.parse_args()

    preprocess_dataset(args.data_dir, args.gt_dir, args.output_dir, args.num_workers)
