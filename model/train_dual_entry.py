"""
Dual Network Joint Training 入口脚本。

两个网络 A 和 B 同时训练，加入 containment loss 确保
P_A(class_a) <= P_B(class_b)（包含约束）。

对标 nnUNet run_training_dual_containment。

使用方式:
    python train_dual_entry.py \
        --preprocessed_dir ./output/preprocessed \
        --work_name dual_exp_01 \
        --batch_size 8 --epochs 1000 \
        --warmup_epochs 500 --lambda_containment 0.2
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

from unet3d import UNet3D, UNetPlusPlus3D
from load_data import StentDatasetV2
from augmentation import get_default_augmentation
from loss import CombinedLoss
from network_modules import (
    AnisoStentUNet,
    containment_loss,
    train_dual,
)


def build_model(arch: str, aniso_stent: bool, device: str):
    """构建单个网络。"""
    if arch == "unet3d":
        base = UNet3D(in_channels=1, out_channels=1, base_channels=8)
    elif arch == "unetpp":
        base = UNetPlusPlus3D(in_channels=1, out_channels=1, base_channels=8)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if aniso_stent:
        return AnisoStentUNet(base, in_channels=1, num_classes=1).to(device)
    return base.to(device)


def build_loaders(preprocessed_dir: str, patch_size: tuple,
                  batch_size: int, num_workers: int,
                  use_augmentation: bool, device: str):
    """构建 train/val DataLoader（两个网络共享同一 DataLoader，保证数据一致）。"""
    dataset = StentDatasetV2(
        preprocessed_dir, patch_size=patch_size,
        oversample_foreground_prob=0.33, transform=None,
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    train_transform = get_default_augmentation('train', patch_size=patch_size) \
        if use_augmentation else None
    val_transform = get_default_augmentation('val', patch_size=patch_size)

    train_dataset = torch.utils.data.Subset(
        StentDatasetV2(
            preprocessed_dir, patch_size=patch_size,
            oversample_foreground_prob=0.33, transform=train_transform,
        ), train_indices)
    val_dataset = torch.utils.data.Subset(
        StentDatasetV2(
            preprocessed_dir, patch_size=patch_size,
            oversample_foreground_prob=0.33, transform=val_transform,
        ), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=(device == "cuda"))
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Dual Network Joint Training")

    # 数据
    parser.add_argument("--preprocessed_dir", type=str, required=True,
                        help="预处理后的 .npy 数据目录")
    parser.add_argument("--work_name", type=str, required=True,
                        help="实验名称")

    # 网络架构
    parser.add_argument("--arch_a", type=str, default="unet3d",
                        choices=["unet3d", "unetpp"],
                        help="网络 A 的架构")
    parser.add_argument("--arch_b", type=str, default="unet3d",
                        choices=["unet3d", "unetpp"],
                        help="网络 B 的架构")
    parser.add_argument("--aniso_stent_a", action='store_true', default=False,
                        help="网络 A 使用 AnisoStentUNet 包装")
    parser.add_argument("--aniso_stent_b", action='store_true', default=False,
                        help="网络 B 使用 AnisoStentUNet 包装")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augmentation", action='store_true', default=False)

    # 包含约束参数
    parser.add_argument("--warmup_epochs", type=int, default=500,
                        help="独立训练轮数（warmup 阶段）")
    parser.add_argument("--lambda_containment", type=float, default=0.2,
                        help="containment loss 最大权重")
    parser.add_argument("--ramp_epochs", type=int, default=50,
                        help="containment 权重的线性爬坡轮数")
    parser.add_argument("--containment_margin", type=float, default=0.0,
                        help="包含约束的容忍边距")
    parser.add_argument("--class_idx_a", type=int, default=1,
                        help="网络 A 中被约束的类别索引")
    parser.add_argument("--class_idx_b", type=int, default=1,
                        help="网络 B 中边界类别的索引")

    # 恢复训练
    parser.add_argument("--checkpoint_a", type=str, default=None)
    parser.add_argument("--checkpoint_b", type=str, default=None)

    # 保存
    parser.add_argument("--save_dir", type=str, default="./output")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    patch_size = (128, 128, 128)

    # 构建数据加载器（两个网络共享同一 DataLoader，确保 case/patch/增强完全一致）
    train_loader, val_loader = build_loaders(
        args.preprocessed_dir, patch_size, args.batch_size,
        args.num_workers, args.augmentation, device)

    # 构建网络
    model_a = build_model(args.arch_a, args.aniso_stent_a, device)
    model_b = build_model(args.arch_b, args.aniso_stent_b, device)

    print(f"Model A: {args.arch_a} (AnisoStent={args.aniso_stent_a}) "
          f"params={sum(p.numel() for p in model_a.parameters()):,}")
    print(f"Model B: {args.arch_b} (AnisoStent={args.aniso_stent_b}) "
          f"params={sum(p.numel() for p in model_b.parameters()):,}")

    # 损失 & 优化器
    criterion_a = CombinedLoss(dice_weight=1.0, lap_weight=0.1).to(device)
    criterion_b = CombinedLoss(dice_weight=1.0, lap_weight=0.1).to(device)

    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999))
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999))

    # 保存目录
    save_dir_a = os.path.join(args.save_dir, args.work_name, "model_a")
    save_dir_b = os.path.join(args.save_dir, args.work_name, "model_b")

    print(f"Dual training: {args.epochs} epochs, warmup={args.warmup_epochs}, "
          f"lambda={args.lambda_containment}, ramp={args.ramp_epochs}")
    print(f"Save dirs: {save_dir_a}, {save_dir_b}")

    # 启动联合训练
    model_a, model_b = train_dual(
        model_a=model_a,
        model_b=model_b,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_a=criterion_a,
        criterion_b=criterion_b,
        optimizer_a=optimizer_a,
        optimizer_b=optimizer_b,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lambda_containment=args.lambda_containment,
        ramp_epochs=args.ramp_epochs,
        containment_margin=args.containment_margin,
        class_idx_a=args.class_idx_a,
        class_idx_b=args.class_idx_b,
        device=device,
        save_dir_a=save_dir_a,
        save_dir_b=save_dir_b,
        checkpoint_path_a=args.checkpoint_a,
        checkpoint_path_b=args.checkpoint_b,
    )

    print("Dual training complete.")


if __name__ == "__main__":
    main()
