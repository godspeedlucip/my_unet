import os
join = os.path.join
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from load_data import StentDataset, StentDatasetV2
from augmentation import get_default_augmentation
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ====== 引入网络 & 损失 ======
from unet3d import UNet3D, UNetPlusPlus3D

from loss import CombinedLoss
from network_modules import (
    AnisoStentUNet,
    AnisotropicAttention,
    TubularEnhancement,
    ProjectionConsistencyLoss,
    CombinedSegProjLoss,
    GenSurfLoss,
    CombinedDTMLoss,
    get_dtm,
)
import time


class PolyLRScheduler:
    """Poly learning rate scheduler, identical to nnUNet's polylr."""
    def __init__(self, optimizer, initial_lr, num_epochs, exponent=0.9):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.num_epochs = num_epochs
        self.exponent = exponent

    def step(self, epoch):
        lr = self.initial_lr * (1 - epoch / self.num_epochs) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ====== Dice指标 ======
def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)

def save_checkpoint(epoch, model, optimizer, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)

def resume_model(checkpoint_path,model,optimizer,device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1 # 从下一轮开始
    print(f"Resuming training from epoch {start_epoch}")
    return model,optimizer,start_epoch

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


def _build_partial_aniso(backbone, aniso_attn=False, tub_enh=False):
    """构建 PartialAnisoUNet 并打印配置。"""
    parts = []
    if aniso_attn:
        parts.append("AnisoAttn")
    if tub_enh:
        parts.append("TubEnh")
    print(f"Using PartialAnisoUNet ({' + '.join(parts)} + {backbone.__class__.__name__})")
    return _PartialAnisoUNet(backbone, in_channels=1, num_classes=1,
                             aniso_attn=aniso_attn, tub_enh=tub_enh)


def compute_dtm_for_batch(seg_batch, voxel_spacing=(1.0, 1.0, 1.0)):
    """为 batch 中每个样本计算前景 DTM，返回 (B, 1, D, H, W) tensor。"""
    import numpy as np
    dtm_list = []
    for i in range(seg_batch.shape[0]):
        seg_np = seg_batch[i, 0].cpu().numpy().astype(np.int16)
        dtm_np = get_dtm(seg_np, voxel_spacing=voxel_spacing, label_list=[0, 1])
        dtm_tensor = torch.from_numpy(dtm_np[..., 1:2].copy()).float()
        dtm_list.append(dtm_tensor.permute(2, 0, 1).unsqueeze(0))  # (D,H,W,1) → (1,1,D,H,W)
    return torch.cat(dtm_list, dim=0)

# ====== 训练循环 ======
def train_model(num_epochs, work_name, proj_path=None, gt_path=None,
                 preprocessed_dir=None, save_dir='./output', batch_size=16,
                 lr=1e-2, device="cuda", checkpoint_path=None,
                 num_workers=4, use_augmentation=False,
                 arch="unet3d", aniso_stent=False,
                 aniso_attn=False, tub_enh=False,
                 proj_consistency=False,
                 proj_weight=0.1, proj_angles="0,90",
                 dtm_loss=False, dtm_weight=0.5, dtm_warmup=250,
                 num_iterations_per_epoch=250):
    # Reduce CPU thread contention during GPU training
    if device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    # 数据集划分：训练 / 验证 / 测试
    # proj_dir = '/data/liuyang/stent/DTR/data/120_120'
    save_path = join(save_dir,work_name)
    model_save_path = join(save_path,'models')
    log_file_path = join(save_path,'training_log.log')
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(model_save_path,exist_ok=True)
    writer = SummaryWriter(join(save_path,'tensorboard'))
    with open(log_file_path, 'a') as f:
        log_line = f"{num_epochs},{work_name},{proj_path},{gt_path},{batch_size},{checkpoint_path}\n"
        f.write(log_line)

    patch_size = (128, 128, 128)

    if preprocessed_dir is not None:
        # 使用预处理后的 .npy 数据（快速路径）
        # 先创建一个无增强的 dataset 来做 train/val split
        dataset = StentDatasetV2(
            preprocessed_dir, patch_size=patch_size,
            oversample_foreground_prob=0.33, transform=None
        )
    else:
        # 回退到原始的 .nii 加载方式
        assert proj_path is not None and gt_path is not None, \
            "需要提供 --data_path 和 --gt_path，或 --preprocessed_dir"
        dataset = StentDataset(proj_path, gt_path, patch_size=patch_size)

    train_size = int(0.9 * len(dataset))  # 选多一点的数据来训练
    val_size = len(dataset) - train_size
    test_size = 0
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 如果使用新数据管线，用 Subset 为 train/val 分别设置不同的 transform
    if preprocessed_dir is not None:
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices

        train_transform = get_default_augmentation('train', patch_size=patch_size) if use_augmentation else None
        val_transform = get_default_augmentation('val', patch_size=patch_size)

        train_dataset = torch.utils.data.Subset(
            StentDatasetV2(
                preprocessed_dir, patch_size=patch_size,
                oversample_foreground_prob=0.33, transform=train_transform
            ),
            train_indices
        )
        val_dataset = torch.utils.data.Subset(
            StentDatasetV2(
                preprocessed_dir, patch_size=patch_size,
                oversample_foreground_prob=0.33, transform=val_transform
            ),
            val_indices
        )

    # DataLoader — use cyclic sampling to match nnUNet fixed-iterations paradigm
    print(f"epochs={num_epochs} work={work_name} preprocessed={preprocessed_dir} "
          f"batch={batch_size} lr={lr} device={device} workers={num_workers} "
          f"aug={use_augmentation} checkpoint={checkpoint_path} "
          f"iters_per_epoch={num_iterations_per_epoch}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device == "cuda"))

    # --- 模型 ---
    if arch == "unetpp":
        backbone = UNetPlusPlus3D(in_channels=1, out_channels=1, base_channels=8)
    else:
        backbone = UNet3D(in_channels=1, out_channels=1, base_channels=8)

    if aniso_stent:
        model = AnisoStentUNet(backbone, in_channels=1, num_classes=1).to(device)
        print(f"Using AnisoStentUNet (AnisoAttn + {arch.upper()} + TubEnh)")
    elif aniso_attn or tub_enh:
        model = _build_partial_aniso(backbone.to(device), aniso_attn=aniso_attn,
                                     tub_enh=tub_enh)
    else:
        model = backbone.to(device)
        print(f"Using {arch.upper()}")

    # torch.compile for free speedup (auto-disabled on Windows/CPU)
    if device == "cuda" and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without it")

    seg_loss = CombinedLoss(dice_weight=1.0, lap_weight=0.1).to(device)
    if proj_consistency:
        angles = [float(a.strip()) for a in proj_angles.split(",")]
        proj_loss_fn = ProjectionConsistencyLoss(angles_deg=angles).to(device)
        criterion = CombinedSegProjLoss(
            seg_loss=seg_loss, proj_loss=proj_loss_fn,
            proj_weight=proj_weight, use_softmax=False,
        ).to(device)
        print(f"Using CombinedSegProjLoss (proj_weight={proj_weight}, angles={angles})")
    elif dtm_loss:
        gsl = GenSurfLoss().to(device)
        criterion = CombinedDTMLoss(
            seg_loss=seg_loss, gsl_loss=gsl,
            gsl_weight=dtm_weight, gsl_warmup_epochs=dtm_warmup,
        ).to(device)
        print(f"Using CombinedDTMLoss (weight={dtm_weight}, warmup={dtm_warmup})")
    else:
        criterion = seg_loss

    # SGD + Nesterov momentum (like nnUNet) — much less VRAM than Adam
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99,
                          weight_decay=3e-5, nesterov=True)
    scheduler = PolyLRScheduler(optimizer, lr, num_epochs, exponent=0.9)
    grad_scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    # best_loss = 1e+20

    start_epoch = 0
    if checkpoint_path is not None:
        model, optimizer, start_epoch = resume_model(checkpoint_path, model, optimizer, device)

    # --- 训练 & 验证 ---
    best_val_dice = -1.0
    print('start training...')
    pbar = tqdm(range(start_epoch, num_epochs), desc='Epoch: ')
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        current_lr = scheduler.step(epoch)

        # --- 训练阶段 (固定迭代次数, 对标 nnUNet) ---
        model.train()
        train_loss, train_dice = 0.0, 0.0
        train_gsl = 0.0
        train_iter = 0
        for inputs, targets in train_loader:
            if train_iter >= num_iterations_per_epoch:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if grad_scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if dtm_loss:
                        dtm_batch = compute_dtm_for_batch(targets).to(device)
                        total, l_seg, l_gsl, alpha = criterion(
                            outputs, targets, dtm_batch, None, epoch)
                    else:
                        total = criterion(outputs, targets)
                grad_scaler.scale(total).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                outputs = model(inputs)
                if dtm_loss:
                    dtm_batch = compute_dtm_for_batch(targets).to(device)
                    total, l_seg, l_gsl, alpha = criterion(
                        outputs, targets, dtm_batch, None, epoch)
                    train_gsl += l_gsl.item()
                else:
                    total = criterion(outputs, targets)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()

            train_loss += total.item()
            train_dice += dice_score(outputs, targets).item()
            train_iter += 1

        avg_train_loss = train_loss / max(train_iter, 1)
        avg_train_dice = train_dice / max(train_iter, 1)

        save_checkpoint(epoch, model, optimizer,
                        os.path.join(model_save_path, "checkpoint_latest.pth"))

        # --- 验证阶段 (固定迭代次数) ---
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        val_gsl = 0.0
        val_iter = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                if val_iter >= num_iterations_per_epoch // 5:
                    break

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if grad_scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        if dtm_loss:
                            dtm_batch = compute_dtm_for_batch(targets).to(device)
                            total, l_seg, l_gsl, alpha = criterion(
                                outputs, targets, dtm_batch, None, epoch)
                            val_gsl += l_gsl.item()
                        else:
                            total = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    if dtm_loss:
                        dtm_batch = compute_dtm_for_batch(targets).to(device)
                        total, l_seg, l_gsl, alpha = criterion(
                            outputs, targets, dtm_batch, None, epoch)
                        val_gsl += l_gsl.item()
                    else:
                        total = criterion(outputs, targets)

                val_loss += total.item()
                val_dice += dice_score(outputs, targets).item()
                val_iter += 1

        avg_val_loss = val_loss / max(val_iter, 1)
        avg_val_dice = val_dice / max(val_iter, 1)

        # 第一个参数是标签，第二个是值，第三个是步数（这里用epoch）
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/dice', avg_val_dice, epoch)
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/dice', avg_train_dice, epoch)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_checkpoint(epoch, model, optimizer,
                            os.path.join(model_save_path, "checkpoint_best.pth"))

        end_time = time.time()
        consume_times = end_time-start_time

        if dtm_loss:
            avg_train_gsl = train_gsl / max(train_iter, 1)
            avg_val_gsl = val_gsl / max(val_iter, 1)
            print(f"Epoch [{epoch+1}/{num_epochs}] LR={current_lr:.2e} "
                  f"Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} "
                  f"| Train GSL: {avg_train_gsl:.4f} "
                  f"| Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} "
                  f"| Val GSL: {avg_val_gsl:.4f} "
                  f"  训练时长: {consume_times:.2f} 秒")
            with open(log_file_path, 'a') as f:
                log_line = f"{epoch+1},{avg_train_loss:.4f},{avg_train_dice:.4f},{avg_train_gsl:.4f},{avg_val_loss:.4f},{avg_val_dice:.4f},{avg_val_gsl:.4f},{consume_times:.2f}\n"
                f.write(log_line)
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] LR={current_lr:.2e} "
                  f"Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} "
                  f"| Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}"
                  f"  训练时长: {consume_times:.2f} 秒")

            # 写入日志文件
            with open(log_file_path, 'a') as f:
                log_line = f"{epoch+1},{avg_train_loss:.4f},{avg_train_dice:.4f},{avg_val_loss:.4f},{avg_val_dice:.4f},{consume_times:.2f}\n"
                f.write(log_line)

    save_checkpoint(epoch, model, optimizer,
                    os.path.join(model_save_path, "checkpoint_final.pth"))
    writer.close() # 关闭tensorboard

    return model

# ====== 启动训练 ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D UNet 训练脚本")
    parser.add_argument("--data_path", type=str, default=None,
                        help="原始 .nii 图像目录 (旧管线)")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="原始 .nii 标注目录 (旧管线)")
    parser.add_argument("--preprocessed_dir", type=str, default=None,
                        help="预处理后的 .npy 数据目录 (新管线, 推荐)")
    parser.add_argument("--work_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='./output')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="初始学习率 (默认 1e-2, 对标 nnUNet)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 进程数 (默认 4, 旧管线用 0)")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--arch", type=str, default="unet3d",
                        choices=["unet3d", "unetpp"],
                        help="网络架构 (默认 unet3d)")
    parser.add_argument("--augmentation", action='store_true', default=False,
                        help="是否启用在线数据增强")
    parser.add_argument("--aniso_stent", action='store_true', default=False,
                        help="使用 AnisoStentUNet (AnisoAttn + TubEnh 全包装)")
    parser.add_argument("--aniso_attn", action='store_true', default=False,
                        help="仅前置各向异性注意力 (AnisotropicAttention)")
    parser.add_argument("--tub_enh", action='store_true', default=False,
                        help="仅后置管状结构增强 (TubularEnhancement)")
    parser.add_argument("--proj_consistency", action='store_true', default=False,
                        help="加入投影一致性自监督损失 (ProjectionConsistencyLoss)")
    parser.add_argument("--proj_weight", type=float, default=0.1,
                        help="投影一致性损失的权重 (默认 0.1)")
    parser.add_argument("--proj_angles", type=str, default="0,90",
                        help="投影角度列表，逗号分隔 (默认 \"0,90\")")
    parser.add_argument("--dtm_loss", action='store_true', default=False,
                        help="启用 DTM 表面损失 (GenSurfLoss)")
    parser.add_argument("--dtm_weight", type=float, default=0.5,
                        help="GenSurfLoss 稳定后的权重 (默认 0.5)")
    parser.add_argument("--dtm_warmup", type=int, default=250,
                        help="多少 epoch 后开始加入 GenSurfLoss (默认 250)")
    parser.add_argument("--num_iterations_per_epoch", type=int, default=250,
                        help="每 epoch 训练迭代次数 (默认 250, 对标 nnUNet)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ', device)

    # 验证输入
    if args.preprocessed_dir is None:
        assert args.data_path and args.gt_path, \
            "需要 --preprocessed_dir 或 (--data_path + --gt_path)"
        assert set(os.listdir(args.data_path)) == set(os.listdir(args.gt_path)), \
            "data_path 和 gt_path 中的文件名不一致"

    model = train_model(
        num_epochs=args.epochs,
        work_name=args.work_name,
        proj_path=args.data_path,
        gt_path=args.gt_path,
        preprocessed_dir=args.preprocessed_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        checkpoint_path=args.checkpoint_path,
        num_workers=args.num_workers,
        use_augmentation=args.augmentation,
        arch=args.arch,
        aniso_stent=args.aniso_stent,
        aniso_attn=args.aniso_attn,
        tub_enh=args.tub_enh,
        proj_consistency=args.proj_consistency,
        proj_weight=args.proj_weight,
        proj_angles=args.proj_angles,
        dtm_loss=args.dtm_loss,
        dtm_weight=args.dtm_weight,
        dtm_warmup=args.dtm_warmup,
        num_iterations_per_epoch=args.num_iterations_per_epoch,
    )


