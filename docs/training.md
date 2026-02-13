# Training Guide

## Overview

Training pipeline:
1. Download STPLS3D raw LAS data
2. Prepare data (tile, encode, generate images)
3. Train the U-Net model
4. Evaluate on validation set

## 1. Download Data

```bash
python scripts/download_stpls3d.py
```

Or manually download from [STPLS3D GitHub](https://github.com/meidachen/STPLS3D) and place in `data/stpls3d/raw/`:

```
data/stpls3d/raw/
├── OCCC_points.las
├── RA_points.las
├── USC_points.las
└── WMSC_points.las
```

## 2. Prepare Training Data

```bash
python training/prepare_data.py --raw-dir data/stpls3d/raw --output-dir data/stpls3d
```

This creates:
```
data/stpls3d/
├── las_cut/         # 250m tiles
├── tensors/         # (512, 512, 7) numpy tensors
├── img_features/    # 3-channel feature PNGs (model input)
└── img_class/       # RGB class mask PNGs (ground truth)
```

## 3. Train

```bash
python training/train.py \
    --features-dir data/stpls3d/img_features \
    --masks-dir data/stpls3d/img_class \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --device cuda \
    --checkpoint-dir checkpoints
```

Key parameters:
- **Architecture**: U-Net + ResNet34 encoder (ImageNet pretrained)
- **Loss**: Multiclass Dice Loss
- **Optimizer**: Adam (lr=1e-3)
- **Data split**: 80% train / 20% validation (random, tile-level)

Checkpoints are saved every epoch to `checkpoints/model_epoch_N.pth`.
Training metrics are logged to `checkpoints/metrics_log.csv`.

## 4. Evaluate

```bash
python training/evaluate.py \
    --checkpoint checkpoints/model_epoch_31.pth \
    --features-dir data/stpls3d/img_features \
    --masks-dir data/stpls3d/img_class \
    --device cuda
```

Prints per-class Precision, Recall, IoU, F1 and overall mIoU.

## 5. Use Trained Model

Copy the best checkpoint to `models/`:

```bash
cp checkpoints/model_epoch_31.pth models/model_epoch_31_multiclass.pth
```

Then use for inference (see [Inference Guide](inference.md)).

## Jupyter Notebook

For interactive experimentation, use:
```
training/notebook/visual_lidar_code_training_model.ipynb
```

## Class Taxonomy (STPLS3D)

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Ground | 10 | Motorcycle |
| 1 | Building | 11 | Light Pole |
| 2 | Low Vegetation | 12 | Street Sign |
| 3 | Medium Vegetation | 13 | Clutter |
| 4 | High Vegetation | 14 | Fence |
| 5 | Vehicle | 15 | Road |
| 6 | Truck | 16 | Sidewalk |
| 7 | Aircraft | 17 | Parking Area |
| 8 | Military Vehicle | 18 | Rail |
| 9 | Bike | 19 | Grass |
