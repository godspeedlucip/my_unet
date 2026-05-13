"""
集成测试 & 使用示例：将 nnUNet 网络模块接入现有 UNet3D。

演示:
1. AnisoStentUNet              — 用各向异性注意力 + 管状增强包装 UNet3D
2. ProjectionConsistencyLoss   — 投影一致性自监督损失
3. CombinedSegProjLoss         — 分割损失 + 投影一致性组合
4. containment_loss            — 双网络包含约束
5. GenSurfLoss / CombinedDTMLoss — DTM 表面损失
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from unet3d import UNet3D
from network_modules import (
    AnisoStentUNet,
    ProjectionConsistencyLoss,
    CombinedSegProjLoss,
    containment_loss,
    GenSurfLoss,
    CombinedDTMLoss,
    LinearSchedule,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_aniso_stent_unet():
    print("=" * 60)
    print("1. AnisoStentUNet test")

    base_unet = UNet3D(in_channels=1, out_channels=1, base_channels=8)
    model = AnisoStentUNet(base_unet, in_channels=1, num_classes=1).to(device)

    x = torch.randn(2, 1, 64, 64, 64, device=device)
    y = model(x)

    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    assert x.shape[0] == y.shape[0]
    print()


def test_projection_consistency():
    print("=" * 60)
    print("2. ProjectionConsistencyLoss test")

    loss_fn = ProjectionConsistencyLoss(angles_deg=[0, 45, 90, 135]).to(device)

    pred = torch.randn(2, 1, 32, 32, 32, device=device)
    target = torch.rand(2, 1, 32, 32, 32, device=device)

    loss = loss_fn(pred, target)
    print(f"   pred: {pred.shape}, target: {target.shape}")
    print(f"   loss: {loss.item():.6f}")
    print()


def test_combined_seg_proj_loss():
    print("=" * 60)
    print("3. CombinedSegProjLoss test")

    seg_loss_fn = torch.nn.BCEWithLogitsLoss()
    proj_loss_fn = ProjectionConsistencyLoss(angles_deg=[0, 90]).to(device)
    combined = CombinedSegProjLoss(
        seg_loss=seg_loss_fn, proj_loss=proj_loss_fn,
        proj_weight=0.1, use_softmax=False,
    ).to(device)

    pred = torch.randn(2, 1, 32, 32, 32, device=device, requires_grad=True)
    target = (torch.rand(2, 1, 32, 32, 32, device=device) > 0.5).float()

    loss = combined(pred, target)
    loss.backward()
    assert pred.grad is not None, "gradient should flow back to pred"

    print(f"   pred: {pred.shape}, target: {target.shape}")
    print(f"   combined loss: {loss.item():.6f}")
    print()


def test_containment_loss():
    print("=" * 60)
    print("4. containment_loss test")

    out_a = torch.randn(2, 2, 16, 16, 16, device=device)
    out_b = torch.randn(2, 2, 16, 16, 16, device=device)

    result = containment_loss(out_a, out_b, class_idx_a=1, class_idx_b=1,
                              margin=0.0, use_sigmoid=True)
    print(f"   out_a: {out_a.shape}, out_b: {out_b.shape}")
    print(f"   loss: {result['loss'].item():.6f}")
    print(f"   violation_rate: {result['violation_rate'].item():.4f}")
    print()


def test_dtm_loss():
    print("=" * 60)
    print("5. GenSurfLoss / CombinedDTMLoss test")

    gen_surf = GenSurfLoss().to(device)
    pred = torch.randn(2, 2, 16, 16, 16, device=device, requires_grad=True)
    dtm = torch.randn(2, 1, 16, 16, 16, device=device)
    valid_mask = (torch.rand(2, 1, 16, 16, 16, device=device) > 0.1).float()

    gsl = gen_surf(pred, dtm, valid_mask)
    print(f"   GenSurfLoss: {gsl.item():.6f}")

    # CombinedDTMLoss
    seg_loss = torch.nn.CrossEntropyLoss()
    combined = CombinedDTMLoss(
        seg_loss=seg_loss, gsl_loss=gen_surf,
        gsl_weight=0.5, gsl_warmup_epochs=250,
    ).to(device)

    target_cls = torch.randint(0, 2, (2, 16, 16, 16), device=device)
    total, l_seg, l_gsl, alpha = combined(pred, target_cls, dtm, valid_mask, epoch=300)
    print(f"   CombinedDTMLoss (epoch=300): total={total.item():.4f}, "
          f"seg={l_seg.item():.4f}, gsl={l_gsl.item():.4f}, alpha={alpha}")

    total, l_seg, l_gsl, alpha = combined(pred, target_cls, dtm, valid_mask, epoch=100)
    print(f"   CombinedDTMLoss (epoch=100): total={total.item():.4f}, "
          f"seg={l_seg.item():.4f}, gsl={l_gsl.item():.4f}, alpha={alpha}")

    total.backward()
    print()


def test_backprop():
    print("=" * 60)
    print("6. Joint backprop (AnisoStentUNet + BCE + ProjConsistency + Containment)")

    base_unet_a = UNet3D(in_channels=1, out_channels=1, base_channels=8)
    base_unet_b = UNet3D(in_channels=1, out_channels=1, base_channels=8)

    model_a = AnisoStentUNet(base_unet_a, in_channels=1, num_classes=1).to(device)
    model_b = AnisoStentUNet(base_unet_b, in_channels=1, num_classes=1).to(device)

    proj_loss_fn = ProjectionConsistencyLoss(angles_deg=[0, 90]).to(device)

    x = torch.randn(2, 1, 64, 64, 64, device=device)
    target = torch.rand(2, 1, 64, 64, 64, device=device)

    out_a = model_a(x)
    out_b = model_b(x)

    sup_loss = torch.nn.functional.binary_cross_entropy_with_logits(out_a, target)
    proj_loss = proj_loss_fn(out_a, target)
    cont = containment_loss(out_a, out_b, class_idx_a=0, class_idx_b=0)

    total = sup_loss + 0.1 * proj_loss + 0.2 * cont['loss']
    total.backward()

    grad_norm_a = sum(p.grad.norm() for p in model_a.parameters() if p.grad is not None)
    grad_norm_b = sum(p.grad.norm() for p in model_b.parameters() if p.grad is not None)

    print(f"   sup_loss: {sup_loss.item():.6f}")
    print(f"   proj_loss: {proj_loss.item():.6f}")
    print(f"   cont_loss: {cont['loss'].item():.6f}")
    print(f"   total: {total.item():.6f}")
    print(f"   grad_norm (model_a): {grad_norm_a:.4f}")
    print(f"   grad_norm (model_b): {grad_norm_b:.4f}")
    print()


def test_linear_schedule():
    print("=" * 60)
    print("7. LinearSchedule test")

    sched = LinearSchedule(num_epochs=1000, init_pause=0)
    print(f"   epoch 0: {sched(0):.4f}")
    print(f"   epoch 250: {sched(250):.4f}")
    print(f"   epoch 500: {sched(500):.4f}")
    print(f"   epoch 999: {sched(999):.4f}")
    print()


def show_integration_code():
    print("=" * 60)
    print("8. Integration examples")
    print("""
# ---- 1. AnisoStentUNet: 包装 UNet3D ----
from unet3d import UNet3D
from network_modules import AnisoStentUNet

base_unet = UNet3D(in_channels=1, out_channels=1, base_channels=8)
model = AnisoStentUNet(base_unet, in_channels=1, num_classes=1).to(device)

# ---- 2. 添加投影一致性正则化 ----
from network_modules import ProjectionConsistencyLoss, CombinedSegProjLoss

proj_loss_fn = ProjectionConsistencyLoss(angles_deg=[0, 45, 90, 135])
combined_loss = CombinedSegProjLoss(
    seg_loss=your_seg_loss,
    proj_loss=proj_loss_fn,
    proj_weight=0.1,
)
loss = combined_loss(outputs, targets)

# ---- 3. 添加 DTM 表面损失 ----
from network_modules import GenSurfLoss, CombinedDTMLoss

dtm_loss = CombinedDTMLoss(
    seg_loss=your_seg_loss,
    gsl_weight=0.5, gsl_warmup_epochs=250,
)
total, l_seg, l_gsl, alpha = dtm_loss(output, target, dtm, valid_mask, epoch)
""")


if __name__ == "__main__":
    test_aniso_stent_unet()
    test_projection_consistency()
    test_combined_seg_proj_loss()
    test_containment_loss()
    test_dtm_loss()
    test_backprop()
    test_linear_schedule()
    show_integration_code()
    print("All tests passed!")
