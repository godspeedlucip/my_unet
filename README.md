# 3D UNet Stent Segmentation — 消融实验

基于 nnUNet 框架的 3D 支架分割网络，集成多种新模块的消融实验。

## 模块

| 模块 | 文件 | 作用 |
|------|------|------|
| AnisotropicAttention | `network_modules/anisotropic_attention.py` | 各向异性伪影去除（前置残差模块） |
| TubularEnhancement | `network_modules/tubular_enhancement.py` | 管状结构增强（后置残差模块） |
| ProjectionConsistency | `network_modules/projection_consistency.py` | 投影一致性自监督损失 |
| DTM Loss | `network_modules/dtm_loss.py` | 距离变换图表面损失 |
| Dual Training | `network_modules/dual_train.py` | 双网络包含约束联合训练 |

## 快速开始

```bash
cd model

# 路径设置
RAW_DIR="/d/desktop/nnUNet/data/nnUNet_raw/Dataset906_Stent"
PREPROCESSED_DIR="/d/desktop/unet_to_improve/output/preprocessed"
OUTPUT_DIR="/d/desktop/unet_to_improve/output"
COMMON="--batch_size 8 --epochs 1000 --lr 1e-4 --augmentation --num_workers 4"

# Step 1: 预处理（只需一次）
python preprocess_v2.py \
    --data_dir "$RAW_DIR/imagesTr" \
    --gt_dir "$RAW_DIR/labelsTr" \
    --output_dir "$PREPROCESSED_DIR" \
    --num_workers 8

# Step 2: 训练（以 AnisoStent 为例）
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_aniso" \
    $COMMON --aniso_stent

# Step 3: 推理
python test.py --nifti --aniso_stent \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_aniso/models/checkpoint_best.pth" \
    --work_name "infer_aniso" \
    --save_dir "$OUTPUT_DIR/inference"
```

完整消融实验命令见 [`bash.md`](bash.md)。

## 文件结构

```
unet/
├── README.md
├── .gitignore
├── bash.md                        # 完整 Pipeline 命令
└── model/
    ├── preprocess_v2.py           # 预处理 .nii.gz → .npy + .pkl
    ├── train_unet.py              # 主训练脚本
    ├── train_dual_entry.py        # 双网络联合训练入口
    ├── test.py                    # 滑窗推理
    ├── unet3d.py                  # UNet3D / UNetPlusPlus3D
    ├── loss.py                    # Dice + Laplacian 损失
    ├── augmentation.py            # 在线数据增强
    ├── load_data.py               # 数据加载
    └── network_modules/           # 消融模块
        ├── anisotropic_attention.py
        ├── tubular_enhancement.py
        ├── aniso_stent_unet.py
        ├── projection_consistency.py
        ├── dtm_loss.py
        ├── dual_train.py
        ├── integration_test.py
        └── __init__.py
```

## Checkpoint

训练只保留三个 checkpoint：

- `checkpoint_best.pth` — 最佳 val dice
- `checkpoint_latest.pth` — 最新 epoch（用于断点恢复）
- `checkpoint_final.pth` — 最终 epoch

恢复训练：

```bash
python train_unet.py ... --checkpoint_path "$OUTPUT_DIR/.../checkpoint_latest.pth"
```
