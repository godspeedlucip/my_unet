"""
双网络联合训练，对标 nnUNet run_training_dual_containment。

核心思路:
- 两个网络 A 和 B 同时训练（例如 A=重建网络输出, B=分割网络输出）
- Containment loss: 网络 A 的 class_a 预测必须被网络 B 的 class_b 预测包含
  即 P_A(class_a) <= P_B(class_b) 对所有体素成立

数据配对保证（关键）:
- 两个网络看到完全相同的输入数据（相同 case、相同 patch 位置、相同增强）
- 通过共享单个 DataLoader 实现，消除 zip(loader_a, loader_b) 的三层不一致

提供的接口:
- containment_loss()       — 包含约束损失
- train_dual()             — 双网络联合训练主循环（适配 ./unet 的现有结构）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time
from typing import Optional, Dict


def containment_loss(
    output_a, output_b,
    class_idx_a: int = 1,
    class_idx_b: int = 1,
    margin: float = 0.0,
    use_sigmoid: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    包含约束: 要求网络 A 的 class_a 预测被网络 B 的 class_b 预测包含。

    对于每个体素: loss = relu(P_A(class_a) - P_B(class_b) - margin)^2

    Args:
        output_a: 网络 A 的输出 (B, C, D, H, W)
        output_b: 网络 B 的输出 (B, C, D, H, W)
        class_idx_a: A 中被约束的类别索引
        class_idx_b: B 中边界类别的索引
        margin: 容忍边距 (A 可以稍大于 B 而不被惩罚)
        use_sigmoid: True=sigmoid, False=softmax

    Returns:
        {'loss': 标量, 'violation_rate': 违法体素比例}
    """
    if use_sigmoid:
        probs_a = torch.sigmoid(output_a)
        probs_b = torch.sigmoid(output_b)
    else:
        probs_a = torch.softmax(output_a, dim=1)
        probs_b = torch.softmax(output_b, dim=1)

    pa = probs_a[:, class_idx_a]
    pb = probs_b[:, class_idx_b]

    violation = torch.relu(pa - pb - margin)
    loss = (violation ** 2).mean()
    violation_rate = (violation > 0).float().mean()

    return {'loss': loss, 'violation_rate': violation_rate}


def train_dual(
    model_a, model_b,
    train_loader,
    val_loader,
    criterion_a, criterion_b,
    optimizer_a, optimizer_b,
    num_epochs: int = 1000,
    warmup_epochs: int = 500,
    lambda_containment: float = 0.2,
    ramp_epochs: int = 50,
    containment_margin: float = 0.0,
    class_idx_a: int = 1,
    class_idx_b: int = 1,
    use_sigmoid_a: bool = True,
    use_sigmoid_b: bool = True,
    device: str = "cuda",
    save_dir_a: str = "./output/model_a",
    save_dir_b: str = "./output/model_b",
    checkpoint_path_a: Optional[str] = None,
    checkpoint_path_b: Optional[str] = None,
):
    """
    双网络联合训练主循环。

    两个网络共享同一个 DataLoader，确保每次迭代看到完全相同的
    (case, patch 位置, 数据增强)，这是 containment loss 有效的前提。

    阶段:
    1. Warmup (epoch 0 ~ warmup_epochs-1): 各自独立训练
    2. Joint  (warmup_epochs ~ num_epochs-1): 加入 containment loss

    Args:
        model_a, model_b: 两个网络
        train_loader: 训练 DataLoader（两个网络共享）
        val_loader: 验证 DataLoader（两个网络共享）
        criterion_a, criterion_b: 各自的监督损失
        optimizer_a, optimizer_b: 各自的优化器
        num_epochs: 总训练轮数
        warmup_epochs: 独立训练轮数（warmup 阶段）
        lambda_containment: containment loss 最大权重
        ramp_epochs: warmup 后 containment 权重的线性爬坡轮数
        containment_margin: 包含约束的容忍边距
        class_idx_a, class_idx_b: 被约束的类别索引
        use_sigmoid_a, use_sigmoid_b: 各自的激活函数
        device: 训练设备
        save_dir_a, save_dir_b: 模型保存目录
        checkpoint_path_a, checkpoint_path_b: 恢复训练的 checkpoint 路径
    """
    import os

    os.makedirs(save_dir_a, exist_ok=True)
    os.makedirs(save_dir_b, exist_ok=True)

    model_a = model_a.to(device)
    model_b = model_b.to(device)

    start_epoch = 0
    if checkpoint_path_a and checkpoint_path_b:
        ckpt_a = torch.load(checkpoint_path_a, map_location=device)
        ckpt_b = torch.load(checkpoint_path_b, map_location=device)
        model_a.load_state_dict(ckpt_a['model_state_dict'])
        model_b.load_state_dict(ckpt_b['model_state_dict'])
        optimizer_a.load_state_dict(ckpt_a['optimizer_state_dict'])
        optimizer_b.load_state_dict(ckpt_b['optimizer_state_dict'])
        start_epoch = ckpt_a['epoch'] + 1
        print(f"Resuming dual training from epoch {start_epoch}")

    def containment_weight(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return lambda_containment
        progress = (epoch - warmup_epochs + 1) / float(ramp_epochs)
        return lambda_containment * max(0.0, min(1.0, progress))

    print(f"Dual training: {num_epochs} epochs, warmup={warmup_epochs}, "
          f"lambda={lambda_containment}, ramp={ramp_epochs}")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        epoch_t0 = time.time()
        lam = containment_weight(epoch)
        phase = "warmup" if epoch < warmup_epochs else "joint"

        # --- 训练阶段 ---
        model_a.train()
        model_b.train()
        tr_loss_a = tr_loss_b = tr_contain = tr_violation = 0.0
        n_batches = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]")
        for inputs, targets in pbar:
            # 同一份数据同时喂给两个网络
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer_a.zero_grad()
            optimizer_b.zero_grad()

            out_a = model_a(inputs)
            out_b = model_b(inputs)

            loss_sup_a = criterion_a(out_a, targets)
            loss_sup_b = criterion_b(out_b, targets)
            cont = containment_loss(out_a, out_b,
                                    class_idx_a=class_idx_a,
                                    class_idx_b=class_idx_b,
                                    margin=containment_margin,
                                    use_sigmoid=use_sigmoid_a)
            loss = loss_sup_a + loss_sup_b + lam * cont['loss']

            loss.backward()
            optimizer_a.step()
            optimizer_b.step()

            tr_loss_a += loss_sup_a.item()
            tr_loss_b += loss_sup_b.item()
            tr_contain += cont['loss'].item()
            tr_violation += cont['violation_rate'].item()
            n_batches += 1

            pbar.set_postfix({
                'A': f'{loss_sup_a.item():.4f}',
                'B': f'{loss_sup_b.item():.4f}',
                'cont': f'{cont["loss"].item():.4f}',
                'lam': f'{lam:.4f}',
            })

        # --- 验证阶段 ---
        model_a.eval()
        model_b.eval()
        val_loss_a = val_loss_b = val_contain = val_violation = 0.0
        n_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                out_a = model_a(inputs)
                out_b = model_b(inputs)

                loss_sup_a = criterion_a(out_a, targets)
                loss_sup_b = criterion_b(out_b, targets)
                cont = containment_loss(out_a, out_b,
                                        class_idx_a=class_idx_a,
                                        class_idx_b=class_idx_b,
                                        margin=containment_margin,
                                        use_sigmoid=use_sigmoid_a)

                val_loss_a += loss_sup_a.item()
                val_loss_b += loss_sup_b.item()
                val_contain += cont['loss'].item()
                val_violation += cont['violation_rate'].item()
                n_val += 1

        # --- 日志 ---
        epoch_time = time.time() - epoch_t0
        print(f"Epoch {epoch+1}/{num_epochs} [{phase}] "
              f"lambda={lam:.4f} | "
              f"Train A:{tr_loss_a/n_batches:.4f} B:{tr_loss_b/n_batches:.4f} "
              f"Cont:{tr_contain/n_batches:.4f} Viol:{tr_violation/n_batches:.4f} | "
              f"Val A:{val_loss_a/n_val:.4f} B:{val_loss_b/n_val:.4f} "
              f"Cont:{val_contain/n_val:.4f} | "
              f"{epoch_time:.1f}s")

        # --- 保存 checkpoint ---
        for model, opt, save_dir, name in [
            (model_a, optimizer_a, save_dir_a, "A"),
            (model_b, optimizer_b, save_dir_b, "B"),
        ]:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, os.path.join(save_dir, f"checkpoint_latest_net{name}.pth"))

        combined_val = val_loss_a + val_loss_b
        if combined_val < best_val_loss:
            best_val_loss = combined_val
            for model, opt, save_dir, name in [
                (model_a, optimizer_a, save_dir_a, "A"),
                (model_b, optimizer_b, save_dir_b, "B"),
            ]:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, os.path.join(save_dir, f"checkpoint_best_net{name}.pth"))

    # --- 保存 final ---
    for model, opt, save_dir, name in [
        (model_a, optimizer_a, save_dir_a, "A"),
        (model_b, optimizer_b, save_dir_b, "B"),
    ]:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, os.path.join(save_dir, f"checkpoint_final_net{name}.pth"))

    return model_a, model_b
