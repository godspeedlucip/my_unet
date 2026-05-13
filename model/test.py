import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
join = os.path.join
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from load_data import ArrayPadder
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ====== 引入网络 & 损失 ======
from unet3d import UNet3D
from network_modules import AnisoStentUNet, AnisotropicAttention, TubularEnhancement
import tifffile as tiff
import nibabel as nib
import tigre
from loss import CombinedLoss


def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)


# def resume_model(checkpoint_path,model,optimizer,device):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1 # 从下一轮开始
#     print(f"Resuming training from epoch {start_epoch}")
#     return model,optimizer,start_epoch

def load_volume(data_path, device="cuda", patch_size=(128,128,128)):
    """
    加载 3D 数据，支持 .tif/.tiff 和 .nii/.nii.gz 格式。
    返回 (1,1,D,H,W) tensor，必要时 pad 到 patch_size 的倍数。
    """
    ext = os.path.splitext(data_path)[-1]
    is_nifti = ext in ('.nii', '.gz') or data_path.endswith('.nii.gz')

    if is_nifti:
        data_nib = nib.load(data_path)
        data = data_nib.get_fdata().astype(np.float32)
        affine, header = data_nib.affine, data_nib.header
    else:
        data = tiff.imread(data_path).astype(np.float32)
        affine, header = None, None

    # Pad 到 patch_size 的整数倍
    padder = ArrayPadder()
    d, h, w = data.shape
    pd, ph, pw = patch_size
    target_d = ((d + pd - 1) // pd) * pd
    target_h = ((h + ph - 1) // ph) * ph
    target_w = ((w + pw - 1) // pw) * pw
    data_pad = padder.pad_to_target_size_diffDimen(
        data, target_size=(target_d, target_h, target_w),
        padding_value=np.min(data))

    tensor = torch.from_numpy(data_pad).float().unsqueeze(0).unsqueeze(0)
    return tensor.to(device), affine, header, data_pad.shape


def sliding_window_inference(model, volume, patch_size=(128,128,128), thred=0.5):
    """
    滑窗推理：将大 volume 拆成 patch_size 的块，逐块推理后拼接。
    """
    _, _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    overlap = (pd // 4, ph // 4, pw // 4)  # 25% overlap

    # 输出 accumulator + weight (用于重叠区域平均)
    output = torch.zeros(1, 1, D, H, W, device=volume.device)
    weight = torch.zeros(1, 1, D, H, W, device=volume.device)

    # 高斯权重（中心高，边缘低）
    gauss_d = torch.exp(-((torch.arange(pd, device=volume.device).float() - pd/2) ** 2) / (2 * (pd/4) ** 2))
    gauss_h = torch.exp(-((torch.arange(ph, device=volume.device).float() - ph/2) ** 2) / (2 * (ph/4) ** 2))
    gauss_w = torch.exp(-((torch.arange(pw, device=volume.device).float() - pw/2) ** 2) / (2 * (pw/4) ** 2))
    gauss = gauss_d[:, None, None] * gauss_h[None, :, None] * gauss_w[None, None, :]

    stride_d = pd - overlap[0]
    stride_h = ph - overlap[1]
    stride_w = pw - overlap[2]

    z_starts = list(range(0, D - pd + 1, stride_d))
    if z_starts and z_starts[-1] + pd < D:
        z_starts.append(D - pd)
    if not z_starts:
        z_starts = [0]

    y_starts = list(range(0, H - ph + 1, stride_h))
    if y_starts and y_starts[-1] + ph < H:
        y_starts.append(H - ph)
    if not y_starts:
        y_starts = [0]

    x_starts = list(range(0, W - pw + 1, stride_w))
    if x_starts and x_starts[-1] + pw < W:
        x_starts.append(W - pw)
    if not x_starts:
        x_starts = [0]

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = volume[:, :, z:z+pd, y:y+ph, x:x+pw]
                with torch.no_grad():
                    pred = torch.sigmoid(model(patch))
                output[:, :, z:z+pd, y:y+ph, x:x+pw] += pred * gauss
                weight[:, :, z:z+pd, y:y+ph, x:x+pw] += gauss

    output = output / weight.clamp(min=1e-8)
    mask = (output > thred).float()
    return mask
    # Lets create a geometry object
    geo = tigre.geometry()
    # VARIABLE                                   DESCRIPTION                    UNITS
    # -------------------------------------------------------------------------------------
    # Distances
    geo.DSD = 1250  # Distance Source Detector      (mm)
    geo.DSO = 750  # Distance Source Origin        (mm)
    # Detector parameters
    geo.nDetector = np.array([512,512])  # number of pixels              (px)
    geo.dDetector = np.array([0.2, 0.2])  # size of each pixel            (mm)
    geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
    # Image parameters
    geo.nVoxel = np.array([256, 128, 128])  # number of voxels              (vx)
    geo.dVoxel = np.array([0.2, 0.2, 0.2])  # size of each voxel            (mm)
    geo.sVoxel = geo.dVoxel * geo.nVoxel  # total size of the image       (mm)
    
    # Offsets
    geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
    # geo.offOrigin = geo.sVoxel /2 

    geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
    # geo.offDetector = geo.sDetector / 2

    geo.accuracy = 0.5  # Variable to define accuracy of
    geo.COR = 0  # y direction displacement for
    geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
    # geo.mode = "cone"  # Or 'parallel'. Geometry type.
    geo.mode = "cone"  # Or 'parallel'. Geometry type.
    return geo


geo = set_geo()
angles = np.linspace(0,np.pi*2,360)

class _PartialAnisoUNet(nn.Module):
    """按需组合 AnisotropicAttention 和/或 TubularEnhancement 的轻量包装器（推理用）。"""

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


# ====== 测试函数 ======
def test_model(model_path, data_path, gt_path, work_name, save_path, thred=0.5,
               device="cuda", if_project=False, aniso_stent=False,
               aniso_attn=False, tub_enh=False,
               patch_size=(128,128,128), is_nifti=False):
    """
    加载模型并对测试集进行评估。

    Args:
        model_path: 保存的模型路径 (.pth)
        data_path: 测试数据目录
        gt_path: GT 标签目录
        work_name: 实验名称
        save_path: 输出保存路径
        thred: 二值化阈值
        device: 运行设备
        if_project: 是否保存 TIGRE 投影
        aniso_stent: 是否使用 AnisoStentUNet wrapper
        patch_size: 推理 patch 尺寸
        is_nifti: True 表示 .nii/.nii.gz 格式, False 表示 .tif 格式
    """

    print(model_path, data_path, gt_path, work_name, save_path, thred, device, if_project)

    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    base_unet = UNet3D(in_channels=1, out_channels=1, base_channels=8)
    if aniso_stent:
        model = AnisoStentUNet(base_unet, in_channels=1, num_classes=1)
        print("Using AnisoStentUNet wrapper for inference")
    elif aniso_attn or tub_enh:
        model = _PartialAnisoUNet(base_unet, in_channels=1, num_classes=1,
                                  aniso_attn=aniso_attn, tub_enh=tub_enh)
        parts = []
        if aniso_attn: parts.append("AnisoAttn")
        if tub_enh: parts.append("TubEnh")
        print(f"Using PartialAnisoUNet ({' + '.join(parts)}) for inference")
    else:
        model = base_unet
    model = model.to(device)
    criterion = CombinedLoss(dice_weight=1.0, lap_weight=0.1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"正在从 {model_path} 加载模型...")

    model.eval()

    vol_save_path = join(save_path, work_name, 'vol')
    proj_save_path = join(save_path, work_name, 'proj')
    os.makedirs(vol_save_path, exist_ok=True)
    os.makedirs(proj_save_path, exist_ok=True)

    if is_nifti:
        # nifti 格式: 自动匹配 imagesTr 和 labelsTr 文件名
        data_files = sorted([f for f in os.listdir(data_path)
                           if f.endswith(('.nii', '.nii.gz'))])
        gt_files = sorted([f for f in os.listdir(gt_path)
                          if f.endswith(('.nii', '.nii.gz'))])

        # 建立 case_id → gt_filename 映射 (去掉 _0000)
        gt_map = {}
        for f in gt_files:
            case = f.replace('.nii.gz', '').replace('.nii', '')
            gt_map[case] = f

        file_pairs = []
        for f in data_files:
            case = f.replace('.nii.gz', '').replace('.nii', '')
            if case.endswith('_0000'):
                case = case[:-5]
            if case in gt_map:
                file_pairs.append((f, gt_map[case]))
        print(f"匹配到 {len(file_pairs)} 对数据/标签文件")
    else:
        # tif 格式: 文件名一致
        data_files = sorted(os.listdir(data_path))
        file_pairs = [(f, f) for f in data_files]

    geo = set_geo()
    angles = np.linspace(0, np.pi * 2, 360)

    total_loss = 0
    total_dice = 0
    nums = 0
    with torch.no_grad():
        for data_file, gt_file in tqdm(file_pairs, desc="推理中"):
            if is_nifti:
                # 加载完整 volume → 滑窗推理
                vol, _, _, padded_shape = load_volume(
                    join(data_path, data_file), device=device,
                    patch_size=patch_size)
                gt_vol, gt_affine, gt_header, _ = load_volume(
                    join(gt_path, gt_file), device=device,
                    patch_size=patch_size)

                mask = sliding_window_inference(
                    model, vol, patch_size=patch_size, thred=thred)

                # 裁剪回原始尺寸 (去掉 padding)
                gt_data = gt_vol[:, :, :gt_vol.shape[2], :gt_vol.shape[3], :gt_vol.shape[4]]
                mask_out = mask[:, :, :mask.shape[2], :mask.shape[3], :mask.shape[4]]

                # 保存为 .nii.gz
                out_name = data_file.replace('_0000.nii.gz', '_pred.nii.gz').replace('.nii.gz', '_pred.nii.gz')
                nib.save(nib.Nifti1Image(
                    mask_out[0, 0].cpu().numpy().astype(np.float32),
                    gt_affine, gt_header), join(vol_save_path, out_name))

                loss = criterion(mask_out, gt_data)
                dice = dice_score(mask_out, gt_data)
            else:
                # tif 格式: 原始逻辑
                train_data = np.expand_dims(tiff.imread(join(data_path, data_file)), axis=0)
                train_data = np.expand_dims(train_data, axis=0)
                train_data = torch.tensor(train_data).float().to(device)
                gt_data = np.expand_dims(tiff.imread(join(gt_path, gt_file)), axis=0)
                gt_data = np.expand_dims(gt_data, axis=0)
                gt_data = torch.tensor(gt_data).float().to(device)

                output = torch.sigmoid(model(train_data))
                mask = (output > thred)

                loss = criterion(mask.float(), gt_data)
                dice = dice_score(mask.float(), gt_data)

                output_cpu = mask[0].cpu().numpy().squeeze()
                tiff.imwrite(join(vol_save_path, data_file), output_cpu.astype(np.float32))

            total_loss += loss.item()
            total_dice += dice.item()
            nums += 1

            if if_project:
                proj = tigre.Ax(
                    mask[0, 0].cpu().numpy().squeeze().astype(np.float32),
                    geo, angles)
                tiff.imwrite(join(proj_save_path, data_file), proj)

    return total_loss / nums, total_dice / nums


def project(vol,geo,angles):
    projections = tigre.Ax(vol.astype(np.float32), geo, angles)
    return projections

# ====== 启动测试 ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--work_name", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--thred", type=float, default=0.5,
                        help="二值化阈值 (默认 0.5)")
    parser.add_argument("--if_project", action='store_true', default=False,
                        help="保存 TIGRE 投影图")
    parser.add_argument("--aniso_stent", action='store_true', default=False,
                        help="使用 AnisoStentUNet 加载 checkpoint")
    parser.add_argument("--aniso_attn", action='store_true', default=False,
                        help="仅使用各向异性注意力 (AnisotropicAttention)")
    parser.add_argument("--tub_enh", action='store_true', default=False,
                        help="仅使用管状结构增强 (TubularEnhancement)")
    parser.add_argument("--nifti", action='store_true', default=False,
                        help="输入为 .nii/.nii.gz 格式 (nnUNet 数据)")
    parser.add_argument("--patch_size", type=str, default="128,128,128",
                        help="推理 patch 尺寸 (默认 128,128,128)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ', device)

    patch_size = tuple(int(x) for x in args.patch_size.split(","))
    assert len(patch_size) == 3, f"patch_size 应为 D,H,W 格式，得到: {args.patch_size}"

    data_path = args.data_path
    model_path = args.model_path
    work_name = args.work_name
    save_dir = args.save_dir
    gt_path = args.gt_path
    thred = args.thred
    log_file_path = join(save_dir, work_name, 'test_res.log')
    print(data_path, gt_path, model_path, work_name, save_dir)
    total_loss, total_dice = test_model(
        model_path, data_path, gt_path, work_name,
        save_dir, thred=thred, device=device,
        if_project=args.if_project, aniso_stent=args.aniso_stent,
        aniso_attn=args.aniso_attn, tub_enh=args.tub_enh,
        patch_size=patch_size, is_nifti=args.nifti)

    with open(log_file_path, 'w') as f:
        para_line = f"data_path: {data_path}, model_path: {model_path}, work_name: {work_name}, save_dir: {save_dir}, gt_path: {gt_path}, thred: {thred}\n"
        log_line = f"{total_loss:.4f},{total_dice:.4f}\n"
        f.write(para_line)
        f.write(log_line)

    print('total_loss: ',total_loss, ' total_dice: ',total_dice)

