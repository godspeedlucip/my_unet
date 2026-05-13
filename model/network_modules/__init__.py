"""
nnUNet 网络扩展模块集合。

模块列表:
- AnisotropicAttention       — 三路各向异性 3D 卷积 + SE 通道注意力，抑制条纹伪影
- TubularEnhancement         — 4 路膨胀卷积，多尺度精炼管状结构
- AnisoStentUNet             — 包装 backbone: AnisoAttn → UNet → TubEnh
- ProjectionConsistencyLoss  — 可微前向投影 MSE 自监督损失
- CombinedSegProjLoss        — 分割损失 + 投影一致性损失组合
- GenSurfLoss                — 广义表面损失 (DTM-based)
- CombinedDTMLoss            — 分割损失 + DTM 表面损失组合
- containment_loss           — 包含约束: P_A(class_a) ≤ P_B(class_b)
- train_dual                 — 双网络联合训练主循环
"""

from .anisotropic_attention import AnisotropicAttention
from .tubular_enhancement import TubularEnhancement
from .aniso_stent_unet import AnisoStentUNet
from .projection_consistency import (
    build_rotation_matrix_3d,
    differentiable_forward_projection,
    ProjectionConsistencyLoss,
    CombinedSegProjLoss,
)
from .dual_train import (
    containment_loss,
    train_dual,
)
from .dtm_loss import (
    get_dtm,
    calculate_class_percentage,
    GenSurfLoss,
    LinearSchedule,
    CombinedDTMLoss,
)
