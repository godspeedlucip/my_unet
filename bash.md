# 完整 Pipeline: 预处理 → 消融实验训练 → 推理

## 文件结构

```
unet/
├── bash.md                        # 本文件
└── model/
    ├── preprocess_v2.py           # 预处理 .nii.gz → .npy + .pkl
    ├── train_unet.py              # 主训练脚本 (UNet3D / UNet++, 所有消融模块)
    ├── train_dual_entry.py        # 双网络联合训练入口
    ├── test.py                    # 推理脚本 (.tif + .nii.gz 滑窗)
    ├── unet3d.py                  # UNet3D + UNetPlusPlus3D 网络定义
    ├── loss.py                    # Dice + Laplacian 损失
    ├── augmentation.py            # 在线数据增强 (对标 batchgeneratorsv2)
    ├── load_data.py               # 数据加载 (ArrayPadder + StentDataset + StentDatasetV2)
    └── network_modules/
        ├── anisotropic_attention.py   # 各向异性伪影去除模块
        ├── tubular_enhancement.py     # 管状结构增强模块
        ├── aniso_stent_unet.py        # AnisoAttn → UNet → TubEnh 包装器
        ├── projection_consistency.py  # 投影一致性损失
        ├── dtm_loss.py                # DTM 表面损失
        ├── dual_train.py              # 双网络包含约束训练
        ├── integration_test.py        # 模块集成测试
        └── __init__.py
```

---

## 路径常量

```bash
cd /d/desktop/unet_to_improve/unet/model

RAW_DIR="/d/desktop/nnUNet/data/nnUNet_raw/Dataset906_Stent"
PREPROCESSED_DIR="/d/desktop/unet_to_improve/output/preprocessed"
OUTPUT_DIR="/d/desktop/unet_to_improve/output"

# 公共训练参数
COMMON="--batch_size 8 --epochs 1000 --lr 1e-4 --augmentation --num_workers 4"
```

---

## Step 1: 预处理（只需运行一次）

将 nnUNet 格式的 `.nii.gz` 原始数据预处理为 `.npy` + `.pkl`。
自动处理 `_0000` 后缀匹配。

```bash
python preprocess_v2.py \
    --data_dir "$RAW_DIR/imagesTr" \
    --gt_dir "$RAW_DIR/labelsTr" \
    --output_dir "$PREPROCESSED_DIR" \
    --num_workers 8
```

输出: `$PREPROCESSED_DIR/{case_id}_data.npy`, `{case_id}_seg.npy`, `{case_id}.pkl`

---

## Step 2: 消融实验训练

所有实验统一使用 `train_unet.py`。架构通过 `--arch` 指定（默认 `unet3d`，可选 `unetpp`）。

### 2.1 Baseline — 纯 UNet3D

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_baseline" \
    $COMMON
```

### 2.2 AnisoAttn Only — 仅各向异性注意力（消融）

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_aniso_attn" \
    $COMMON \
    --aniso_attn
```

### 2.3 TubEnh Only — 仅管状结构增强（消融）

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_tub_enh" \
    $COMMON \
    --tub_enh
```

### 2.4 AnisoStent — AnisoAttn + TubEnh 全包装

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_aniso" \
    $COMMON \
    --aniso_stent
```

### 2.5 ProjCons — 投影一致性损失

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_proj" \
    $COMMON \
    --proj_consistency --proj_weight 0.1 --proj_angles "0,45,90,135"
```

### 2.6 DTM — 距离变换图表面损失

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_dtm" \
    $COMMON \
    --dtm_loss --dtm_weight 0.5 --dtm_warmup 250
```

### 2.7 AnisoStent + ProjCons

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_aniso_proj" \
    $COMMON \
    --aniso_stent --proj_consistency --proj_weight 0.1 --proj_angles "0,45,90,135"
```

### 2.8 AnisoStent + DTM

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_aniso_dtm" \
    $COMMON \
    --aniso_stent --dtm_loss --dtm_weight 0.5 --dtm_warmup 250
```

### 2.9 ProjCons + DTM

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_proj_dtm" \
    $COMMON \
    --proj_consistency --dtm_loss
```

### 2.10 All — 全部三个模块

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_all" \
    $COMMON \
    --aniso_stent --proj_consistency --dtm_loss
```

### 2.11 Dual Training — 双网络包含约束训练

```bash
python train_dual_entry.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_dual" \
    $COMMON \
    --warmup_epochs 500 --lambda_containment 0.2 --ramp_epochs 50
```

### 2.12 UNet++ Baseline (附加实验)

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_unetpp_baseline" \
    $COMMON \
    --arch unetpp
```

### 2.13 UNet++ + AnisoStent (附加实验)

```bash
python train_unet.py \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --work_name "ablation_unetpp_aniso" \
    $COMMON \
    --arch unetpp --aniso_stent
```

---

## Step 3: 推理

所有实验使用 `test.py --nifti` 进行滑窗推理。
`--aniso_stent` / `--aniso_attn` / `--tub_enh` 标志需与训练时一致。

### 3.1 Baseline

```bash
python test.py --nifti \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_baseline/models/checkpoint_best.pth" \
    --work_name "infer_baseline" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.2 AnisoAttn Only

```bash
python test.py --nifti --aniso_attn \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_aniso_attn/models/checkpoint_best.pth" \
    --work_name "infer_aniso_attn" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.3 TubEnh Only

```bash
python test.py --nifti --tub_enh \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_tub_enh/models/checkpoint_best.pth" \
    --work_name "infer_tub_enh" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.4 AnisoStent

```bash
python test.py --nifti --aniso_stent \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_aniso/models/checkpoint_best.pth" \
    --work_name "infer_aniso" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.5 ProjCons

```bash
python test.py --nifti \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_proj/models/checkpoint_best.pth" \
    --work_name "infer_proj" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.6 DTM

```bash
python test.py --nifti \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_dtm/models/checkpoint_best.pth" \
    --work_name "infer_dtm" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.7 AnisoStent + ProjCons

```bash
python test.py --nifti --aniso_stent \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_aniso_proj/models/checkpoint_best.pth" \
    --work_name "infer_aniso_proj" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.8 AnisoStent + DTM

```bash
python test.py --nifti --aniso_stent \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_aniso_dtm/models/checkpoint_best.pth" \
    --work_name "infer_aniso_dtm" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.9 ProjCons + DTM

```bash
python test.py --nifti \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_proj_dtm/models/checkpoint_best.pth" \
    --work_name "infer_proj_dtm" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.10 All

```bash
python test.py --nifti --aniso_stent \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_all/models/checkpoint_best.pth" \
    --work_name "infer_all" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.11 Dual Training (推理两个网络)

```bash
# 网络 A
python test.py --nifti \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_dual/model_a/checkpoint_best_netA.pth" \
    --work_name "infer_dual_A" \
    --save_dir "$OUTPUT_DIR/inference"

# 网络 B
python test.py --nifti \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_dual/model_b/checkpoint_best_netB.pth" \
    --work_name "infer_dual_B" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.12 UNet++ Baseline (附加实验)

```bash
python test.py --nifti \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_unetpp_baseline/models/checkpoint_best.pth" \
    --work_name "infer_unetpp_baseline" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.13 UNet++ + AnisoStent (附加实验)

```bash
python test.py --nifti --aniso_stent \
    --data_path "$RAW_DIR/imagesTr" \
    --gt_path "$RAW_DIR/labelsTr" \
    --model_path "$OUTPUT_DIR/ablation_unetpp_aniso/models/checkpoint_best.pth" \
    --work_name "infer_unetpp_aniso" \
    --save_dir "$OUTPUT_DIR/inference"
```

### 3.14 批量推理（一键运行所有实验）

```bash
# 需要 --aniso_stent 的实验
ANISO_EXPS="ablation_aniso ablation_aniso_proj ablation_aniso_dtm ablation_all \
            ablation_unetpp_aniso"
# 需要 --aniso_attn 的实验
ANISO_ATTN_EXPS="ablation_aniso_attn"
# 需要 --tub_enh 的实验
TUB_ENH_EXPS="ablation_tub_enh"

for exp in ablation_baseline ablation_aniso_attn ablation_tub_enh \
           ablation_aniso ablation_proj ablation_dtm \
           ablation_aniso_proj ablation_aniso_dtm ablation_proj_dtm ablation_all \
           ablation_unetpp_baseline ablation_unetpp_aniso; do
    FLAGS=""
    for e in $ANISO_EXPS; do
        [ "$exp" = "$e" ] && FLAGS="$FLAGS --aniso_stent"
    done
    for e in $ANISO_ATTN_EXPS; do
        [ "$exp" = "$e" ] && FLAGS="$FLAGS --aniso_attn"
    done
    for e in $TUB_ENH_EXPS; do
        [ "$exp" = "$e" ] && FLAGS="$FLAGS --tub_enh"
    done
    # Dual 模型路径特殊处理
    if [ "$exp" = "ablation_dual" ]; then
        python test.py --nifti \
            --data_path "$RAW_DIR/imagesTr" \
            --gt_path "$RAW_DIR/labelsTr" \
            --model_path "$OUTPUT_DIR/ablation_dual/model_a/checkpoint_best_netA.pth" \
            --work_name "infer_dual_A" \
            --save_dir "$OUTPUT_DIR/inference"
        python test.py --nifti \
            --data_path "$RAW_DIR/imagesTr" \
            --gt_path "$RAW_DIR/labelsTr" \
            --model_path "$OUTPUT_DIR/ablation_dual/model_b/checkpoint_best_netB.pth" \
            --work_name "infer_dual_B" \
            --save_dir "$OUTPUT_DIR/inference"
    else
        python test.py --nifti $FLAGS \
            --data_path "$RAW_DIR/imagesTr" \
            --gt_path "$RAW_DIR/labelsTr" \
            --model_path "$OUTPUT_DIR/${exp}/models/checkpoint_best.pth" \
            --work_name "infer_${exp}" \
            --save_dir "$OUTPUT_DIR/inference"
    fi
done
```

---

## 消融实验对照表

| # | 实验名 | Arch | AnisoAttn | TubEnh | ProjCons | DTM | Dual |
|---|--------|------|:---------:|:------:|:--------:|:---:|:----:|
| 1 | `ablation_baseline` | UNet3D | | | | | |
| 2 | `ablation_aniso_attn` | UNet3D | ✓ | | | | |
| 3 | `ablation_tub_enh` | UNet3D | | ✓ | | | |
| 4 | `ablation_aniso` | UNet3D | ✓ | ✓ | | | |
| 5 | `ablation_proj` | UNet3D | | | ✓ | | |
| 6 | `ablation_dtm` | UNet3D | | | | ✓ | |
| 7 | `ablation_aniso_proj` | UNet3D | ✓ | ✓ | ✓ | | |
| 8 | `ablation_aniso_dtm` | UNet3D | ✓ | ✓ | | ✓ | |
| 9 | `ablation_proj_dtm` | UNet3D | | | ✓ | ✓ | |
| 10 | `ablation_all` | UNet3D | ✓ | ✓ | ✓ | ✓ | |
| 11 | `ablation_dual` | UNet3D×2 | | | | | ✓ |
| 12 | `ablation_unetpp_baseline` | UNet++ | | | | | |
| 13 | `ablation_unetpp_aniso` | UNet++ | ✓ | ✓ | | | |

---

## train_unet.py 全部参数

```
python train_unet.py --help

  --preprocessed_dir PATH   预处理数据目录（推荐）
  --data_path PATH          原始 .nii 图像目录（旧管线）
  --gt_path PATH            原始 .nii 标注目录（旧管线）
  --work_name NAME          实验名称（必填）
  --save_dir PATH           输出根目录（默认 ./output）
  --arch {unet3d,unetpp}    网络架构（默认 unet3d）
  --batch_size N            批次大小（默认 16）
  --epochs N                训练轮数（默认 1000）
  --lr LR                   学习率（默认 1e-4）
  --num_workers N           DataLoader 进程数（默认 4）
  --augmentation            启用在线数据增强
  --checkpoint_path PATH    恢复训练的 checkpoint
  --aniso_stent             使用 AnisoStentUNet 包装 (AnisoAttn + TubEnh)
  --aniso_attn              仅前置各向异性注意力 (AnisotropicAttention)
  --tub_enh                 仅后置管状结构增强 (TubularEnhancement)
  --proj_consistency        启用投影一致性损失
  --proj_weight FLOAT       投影损失权重（默认 0.1）
  --proj_angles STR         投影角度，逗号分隔（默认 "0,90"）
  --dtm_loss                启用 DTM 表面损失
  --dtm_weight FLOAT        DTM 损失权重（默认 0.5）
  --dtm_warmup N            DTM warmup epoch 数（默认 250）
```

---

## 结果查看

```bash
# 训练日志 (TensorBoard)
tensorboard --logdir "$OUTPUT_DIR"

# 推理 Dice / Loss 汇总
for f in "$OUTPUT_DIR/inference"/*/test_res.log; do
    echo "$(dirname $f | xargs basename): $(tail -1 $f)"
done
```
