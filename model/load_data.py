import torch
from torch.utils.data import DataLoader, Dataset
import os
join = os.path.join
import tifffile as tiff
import numpy as np
import random
import nibabel as nib
from scipy.ndimage import uniform_filter



class ArrayPadder: # 假设这个方法在一个类中
    def pad_to_target_size_diffDimen(
        self,
        input_array, 
        target_size=(128, 128, 128),
        padding_value=0
    ) -> np.ndarray:
        """
        将三维 NumPy 数组填充 (Pad) 到目标尺寸。
        
        策略：
        1. 只填充输入尺寸小于目标尺寸的维度，尽可能居中。
        2. 对于输入尺寸大于或等于目标尺寸的维度，不进行填充，也不抛出异常。
           这意味着最终数组的某些维度可能会大于 target_size。

        Args:
            input_array: 输入的三维数组 (D, H, W)。
            target_size: 目标形状 (D_target, H_target, W_target)。
            padding_value: 用于填充的值 (默认为 0)。

        Returns:
            padded_array: 填充后的数组。其形状取决于输入数组和 target_size。
        """
        # 维度检查保持不变
        if input_array.ndim != 3:
            raise ValueError(f"输入数组必须是三维的，但它的维度是 {input_array.ndim}。")
        if len(target_size) != 3:
             raise ValueError(f"目标尺寸必须是三维的 (D, H, W)，但传入的是 {target_size}。")

        # 1. 初始化填充宽度列表
        pad_width = []
        
        # 2. 计算每个维度的填充量
        for input_dim, target_dim in zip(input_array.shape, target_size):
            
            if input_dim < target_dim:
                # 只有当输入小于目标时，才计算填充量
                
                # 需要填充的总量
                total_padding = target_dim - input_dim
                
                # 计算前部填充量 (尽量居中)
                pad_before = total_padding // 2
                
                # 计算后部填充量 (确保总和正确)
                pad_after = total_padding - pad_before
                
                pad_width.append((pad_before, pad_after))
                
            else: 
                # input_dim >= target_dim
                # 保持该维度不变，填充宽度为零
                pad_width.append((0, 0))

        # 3. 执行 np.pad 填充
        padded_array = np.pad(
            input_array, 
            pad_width=pad_width, 
            mode='constant', 
            constant_values=padding_value
        )
        
        return padded_array
    
    def pad_to_target_size_AllDimen(
        self,
        input_array, 
        target_size = (128, 128, 128),
        padding_value = 0
    ) -> np.ndarray:
        """
        将三维 NumPy 数组填充 (Pad) 到目标尺寸，使用零填充。
        填充策略：在每个轴上尽可能居中。

        Args:
            input_array: 输入的三维数组 (D, H, W)。
            target_size: 目标形状 (D_target, H_target, W_target)。
            padding_value: 用于填充的值 (默认为 0)。

        Returns:
            padded_array: 填充后的数组，形状为 target_size。
        """
        if input_array.ndim != 3:
            raise ValueError(f"输入数组必须是三维的，但它的维度是 {input_array.ndim}。")

        # 1. 初始化填充宽度列表
        pad_width = []
        
        # 2. 计算每个维度的填充量
        for input_dim, target_dim in zip(input_array.shape, target_size):
            if input_dim > target_dim:
                raise ValueError(f"输入尺寸 {input_dim} 大于目标尺寸 {target_dim}，无法填充。")

            # 需要填充的总量
            total_padding = target_dim - input_dim
            
            # 计算前部填充量 (尽量居中)
            pad_before = total_padding // 2
            
            # 计算后部填充量 (确保总和正确)
            pad_after = total_padding - pad_before
            
            pad_width.append((pad_before, pad_after))

        # 3. 执行 np.pad 填充
        # mode='constant' 表示使用一个常数值填充
        padded_array = np.pad(
            input_array, 
            pad_width=pad_width, 
            mode='constant', 
            constant_values=padding_value
        )
        
        return padded_array


def extract_max_cover_patch(mask, patch_size=(128, 64, 64)):
    """
    在 3D mask 数据中裁剪一个 patch，最大限度覆盖有效值 (1)。
    
    参数:
        mask: numpy.ndarray, shape=(D, W, H)，0/1 mask
        patch_size: tuple, 需要裁剪的 patch 尺寸 (pd, pw, ph)

    返回:
        patch: numpy.ndarray, 裁剪得到的 patch
        start: tuple, 裁剪区域的起始索引 (d, w, h)
    """
    D, W, H = mask.shape
    pd, pw, ph = patch_size
    
    if pd > D or pw > W or ph > H:
        raise ValueError("Patch 大小超过了输入体素的大小")

    # 用 uniform_filter 高效计算任意窗口内 1 的数量
    # 注意：uniform_filter 给的是平均值，所以要乘以窗口体积得到和
    patch_volume = pd * pw * ph
    summed = uniform_filter(mask.astype(np.float32), size=patch_size, mode="constant") * patch_volume

    # 找到最大值的位置
    max_pos = np.unravel_index(np.argmax(summed), summed.shape)

    # uniform_filter 输出的值是“居中”窗口的平均数，因此需要偏移一半
    d0 = max(0, min(D - pd, max_pos[0] - pd // 2))
    w0 = max(0, min(W - pw, max_pos[1] - pw // 2))
    h0 = max(0, min(H - ph, max_pos[2] - ph // 2))

    patch = mask[d0:d0+pd, w0:w0+pw, h0:h0+ph]
    return patch, (d0, w0, h0)



class StentDataset(Dataset):
    def __init__(self, train_dir, gt_dir,patch_size =(128,128,128)):
        super().__init__()
        self.train_dir = train_dir
        self.gt_dir = gt_dir
        self.all_data = sorted(os.listdir(self.train_dir))
        self.gt_data = sorted(os.listdir(self.gt_dir))
        self.patch_size = patch_size # 新增的裁剪尺寸参数

    def pad_to_target_size(
        self,
        input_array, 
        target_size = (128, 128, 128),
        padding_value = 0
    ) -> np.ndarray:
        """
        将三维 NumPy 数组填充 (Pad) 到目标尺寸，使用零填充。
        填充策略：在每个轴上尽可能居中。

        Args:
            input_array: 输入的三维数组 (D, H, W)。
            target_size: 目标形状 (D_target, H_target, W_target)。
            padding_value: 用于填充的值 (默认为 0)。

        Returns:
            padded_array: 填充后的数组，形状为 target_size。
        """
        if input_array.ndim != 3:
            raise ValueError(f"输入数组必须是三维的，但它的维度是 {input_array.ndim}。")

        # 1. 初始化填充宽度列表
        pad_width = []
        
        # 2. 计算每个维度的填充量
        for input_dim, target_dim in zip(input_array.shape, target_size):
            if input_dim > target_dim:
                raise ValueError(f"输入尺寸 {input_dim} 大于目标尺寸 {target_dim}，无法填充。")

            # 需要填充的总量
            total_padding = target_dim - input_dim
            
            # 计算前部填充量 (尽量居中)
            pad_before = total_padding // 2
            
            # 计算后部填充量 (确保总和正确)
            pad_after = total_padding - pad_before
            
            pad_width.append((pad_before, pad_after))

        # 3. 执行 np.pad 填充
        # mode='constant' 表示使用一个常数值填充
        padded_array = np.pad(
            input_array, 
            pad_width=pad_width, 
            mode='constant', 
            constant_values=padding_value
        )
        
        return padded_array

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # train_data = tiff.imread(join(self.train_dir,self.all_data[idx])).astype(np.float32)
        # gt_data = tiff.imread(join(self.gt_dir,self.gt_data[idx])).astype(np.float32)
        # print(11111)
        train_data = nib.load(join(self.train_dir,self.all_data[idx])).get_fdata().astype(np.float32)
        gt_data = nib.load(join(self.gt_dir,self.gt_data[idx])).get_fdata().astype(np.float32)

        d, h, w = train_data.shape
        p_d, p_h, p_w = self.patch_size
        # print(d, h, w, p_d, p_h, p_w)
        if(d >= p_d):
            start_d = random.randint(0, d - p_d)
        else:
            start_d = 0
            p_d = d
        if(h >= p_h):
            start_h = random.randint(0, h - p_h)
        else:
            start_h = 0
            p_h = h
        if(w >= p_w):
            start_w = random.randint(0, w - p_w)
        else:
            start_w = 0
            p_w = w
        cropped_train = train_data[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]
        cropped_gt = gt_data[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]

        ## 将数据pad到 (128,128,128)
        # print(type(np.min(cropped_gt)))
        cropped_gt = self.pad_to_target_size(cropped_gt, (128,128,128), padding_value=0)
        cropped_train = self.pad_to_target_size(cropped_train,(128,128,128), padding_value=np.min(cropped_train))

        cropped_train = np.expand_dims(cropped_train,axis=0)
        cropped_gt = np.expand_dims(cropped_gt,axis=0)
        return cropped_train, cropped_gt
    

# if __name__ == "__main__":
#     proj_path = '/data1/liuyang/stent/DTR/data/train_data/new/90_4-20-fdkNoFilter/vol_train'
#     gt_path = '/data1/liuyang/stent/DTR/data/train_data/new/90_4-20-fdkNoFilter/vol_gt'
#     dataset = StentDataset(proj_path,gt_path)
#     train_size = int(0.9 * len(dataset)) #选多一点的数据来训练
#     val_size = len(dataset) - train_size
#     test_size = 0
#     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
#         dataset, [train_size, val_size, test_size]
#     )

#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

#     for inputs, targets in train_loader:
#         pass


# ═══════════════════════════════════════════════════════════════════════
# StentDatasetV2 — 快速 mmap 数据加载（推荐）
# ═══════════════════════════════════════════════════════════════════════
import pickle

class StentDatasetV2(torch.utils.data.Dataset):
    """
    从预处理目录加载 .npy 数据的快速 Dataset。

    Args:
        preprocessed_dir: 预处理输出目录（包含 *_data.npy, *_seg.npy, *.pkl）
        patch_size: 裁剪 patch 尺寸 (D, H, W)
        oversample_foreground_prob: 前景过采样比例 (0.0-1.0), nnUNet 默认 0.33
        transform: 数据增强组合 (ComposeTransforms 或 None)
    """

    def __init__(self, preprocessed_dir, patch_size=(128, 128, 128),
                 oversample_foreground_prob=0.33, transform=None):
        super().__init__()

        self.preprocessed_dir = preprocessed_dir
        self.patch_size = np.array(patch_size, dtype=int)
        self.oversample_foreground_prob = oversample_foreground_prob
        self.transform = transform

        # 扫描所有 .pkl 文件找到 case
        self.case_ids = []
        for f in sorted(os.listdir(preprocessed_dir)):
            if f.endswith('.pkl'):
                case_id = f[:-4]
                data_path = os.path.join(preprocessed_dir, f"{case_id}_data.npy")
                seg_path = os.path.join(preprocessed_dir, f"{case_id}_seg.npy")
                if os.path.exists(data_path) and os.path.exists(seg_path):
                    self.case_ids.append(case_id)

        # 预加载所有 .npy 文件（内存映射）和 .pkl 属性
        self.data_mmaps = {}
        self.seg_mmaps = {}
        self.properties = {}

        for case_id in self.case_ids:
            data_path = os.path.join(preprocessed_dir, f"{case_id}_data.npy")
            seg_path = os.path.join(preprocessed_dir, f"{case_id}_seg.npy")
            props_path = os.path.join(preprocessed_dir, f"{case_id}.pkl")

            self.data_mmaps[case_id] = np.load(data_path, mmap_mode='r')
            self.seg_mmaps[case_id] = np.load(seg_path, mmap_mode='r')

            with open(props_path, 'rb') as f:
                self.properties[case_id] = pickle.load(f)

        self._has_class_locations = {}
        for case_id in self.case_ids:
            cls_locs = self.properties[case_id].get('class_locations', {})
            self._has_class_locations[case_id] = len(cls_locs) > 0

    def __len__(self):
        return len(self.case_ids)

    def _get_bbox_from_center(self, center, shape):
        half = self.patch_size // 2
        bbox = []
        need_pad_after = []
        shape_3d = np.array(shape)

        for dim in range(3):
            start = center[dim] - half[dim]
            end = start + self.patch_size[dim]
            pad_before = max(0, -start)
            pad_after = max(0, end - shape_3d[dim])
            start = max(0, start)
            end = min(shape_3d[dim], end)
            bbox.append((start, end))
            need_pad_after.append((pad_before, pad_after))

        return bbox, need_pad_after

    def _extract_patch(self, data_vol, seg_vol, bbox, need_pad):
        data_patch = data_vol[bbox[0][0]:bbox[0][1],
                              bbox[1][0]:bbox[1][1],
                              bbox[2][0]:bbox[2][1]]
        seg_patch = seg_vol[bbox[0][0]:bbox[0][1],
                            bbox[1][0]:bbox[1][1],
                            bbox[2][0]:bbox[2][1]]

        if any(p[0] > 0 or p[1] > 0 for p in need_pad):
            pad_width = [(p[0], p[1]) for p in need_pad]
            data_patch = np.pad(data_patch, pad_width, mode='constant', constant_values=0)
            seg_patch = np.pad(seg_patch, pad_width, mode='constant', constant_values=0)

        return data_patch, seg_patch

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]

        data_vol = self.data_mmaps[case_id]
        seg_vol = self.seg_mmaps[case_id]
        props = self.properties[case_id]
        shape = data_vol.shape

        do_oversample = (
            self._has_class_locations[case_id] and
            np.random.random() < self.oversample_foreground_prob
        )

        if do_oversample:
            cls_locs = props['class_locations']
            class_ids = list(cls_locs.keys())
            chosen_class = class_ids[np.random.randint(len(class_ids))]
            locs = cls_locs[chosen_class]
            center = locs[np.random.randint(len(locs))]
        else:
            half = self.patch_size // 2
            center = np.array([
                np.random.randint(max(0, -half[0]), max(1, shape[0] - half[0]) + 1),
                np.random.randint(max(0, -half[1]), max(1, shape[1] - half[1]) + 1),
                np.random.randint(max(0, -half[2]), max(1, shape[2] - half[2]) + 1),
            ])

        bbox, need_pad = self._get_bbox_from_center(center, shape)
        data_patch, seg_patch = self._extract_patch(data_vol, seg_vol, bbox, need_pad)

        data_patch = np.expand_dims(data_patch, axis=0)

        data_tensor = torch.from_numpy(data_patch).float()
        seg_tensor = torch.from_numpy(seg_patch).float()

        if self.transform is not None:
            data_dict = {'image': data_tensor, 'segmentation': seg_tensor}
            data_dict = self.transform(**data_dict)
            data_tensor = data_dict['image']
            seg_tensor = data_dict['segmentation']

        if seg_tensor.ndim == 3:
            seg_tensor = seg_tensor.unsqueeze(0)

        return data_tensor, seg_tensor
