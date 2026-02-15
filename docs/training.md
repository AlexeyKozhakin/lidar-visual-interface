# Training Guide

## Overview

Training pipeline:
1. Download STPLS3D raw LAS data
2. Prepare data (tile, encode, generate images)
3. Train the U-Net model
4. Evaluate on validation set

## 1. Download Data

Linux/macOS:
```bash
python scripts/download_stpls3d.py
```

Windows PowerShell:
```powershell
python scripts\download_stpls3d.py
```

Windows CMD:
```cmd
python scripts\download_stpls3d.py
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

Linux/macOS:
```bash
python training/prepare_data.py --raw-dir data/stpls3d/raw --output-dir data/stpls3d
```

Windows PowerShell:
```powershell
python training\prepare_data.py --raw-dir data\stpls3d\raw --output-dir data\stpls3d
```

Windows CMD:
```cmd
python training\prepare_data.py --raw-dir data\stpls3d\raw --output-dir data\stpls3d
```

This creates:
```
data/stpls3d/
├── las_cut/         # 250m tiles
├── tensors/         # (512, 512, 7) numpy tensors
├── img_features/    # 3-channel feature PNGs (model input)
├── img_rgb/         # RGB PNGs from LAS RGB channels (visual QA)
└── img_class/       # RGB class mask PNGs (ground truth)
```

If your input is already tiled LAS (for example from `prepare_stpls3d_las.py`), skip re-tiling:

Linux/macOS:
```bash
python -m training.prepare_data \
  --raw-dir data/stpls3d_prepared/tiles_las \
  --output-dir data/stpls3d_train_ready \
  --skip-tiling
```

Windows PowerShell:
```powershell
python -m training.prepare_data `
  --raw-dir data\stpls3d_prepared\tiles_las `
  --output-dir data\stpls3d_train_ready `
  --skip-tiling
```

Windows CMD:
```cmd
python -m training.prepare_data ^
  --raw-dir data\stpls3d_prepared\tiles_las ^
  --output-dir data\stpls3d_train_ready ^
  --skip-tiling
```

### Optional: STPLS3D PLY -> stretched LAS tiles + density reports

If your raw STPLS3D data is in `.ply`, run:

Linux/macOS:
```bash
python training/prepare_stpls3d_las.py \
  --input-dir data/stpls3d/raw_ply \
  --output-dir data/stpls3d_prepared \
  --tile-size 250 \
  --target-points-per-tile 2000000 \
  --seed 42
```

Windows PowerShell:
```powershell
python training\prepare_stpls3d_las.py `
  --input-dir data\stpls3d\raw_ply `
  --output-dir data\stpls3d_prepared `
  --tile-size 250 `
  --target-points-per-tile 2000000 `
  --seed 42
```

Windows CMD:
```cmd
python training\prepare_stpls3d_las.py ^
  --input-dir data\stpls3d\raw_ply ^
  --output-dir data\stpls3d_prepared ^
  --tile-size 250 ^
  --target-points-per-tile 2000000 ^
  --seed 42
```

What this step does:
- Converts each PLY file to LAS (preserving class labels 1:1 and RGB).
- Normalizes coordinates so `min(x)=0` and `min(y)=0`.
- Stretches XY size to the nearest multiple of `tile-size`.
- Splits into `tile-size x tile-size` LAS tiles.
- Downsamples only if a tile is above `target-points-per-tile`.
- Marks low-density tiles (below target) in reports without adding synthetic points.

Outputs:
```
data/stpls3d_prepared/
├── scenes_las/                 # scene-level stretched LAS files
├── tiles_las/                  # per-tile LAS files
├── reports/
│   ├── per_file_report.csv
│   ├── per_file_report.json
│   ├── per_tile_report.csv
│   └── summary.json
└── logs/
    └── prepare.log
```

## 3. Train

Linux/macOS:
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

Windows PowerShell:
```powershell
python training\train.py `
    --features-dir data\stpls3d\img_features `
    --masks-dir data\stpls3d\img_class `
    --epochs 100 `
    --batch-size 8 `
    --lr 0.001 `
    --device cuda `
    --checkpoint-dir checkpoints
```

Windows CMD:
```cmd
python training\train.py ^
    --features-dir data\stpls3d\img_features ^
    --masks-dir data\stpls3d\img_class ^
    --epochs 100 ^
    --batch-size 8 ^
    --lr 0.001 ^
    --device cuda ^
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

Linux/macOS:
```bash
python training/evaluate.py \
    --checkpoint checkpoints/model_epoch_31.pth \
    --features-dir data/stpls3d/img_features \
    --masks-dir data/stpls3d/img_class \
    --device cuda
```

Windows PowerShell:
```powershell
python training\evaluate.py `
    --checkpoint checkpoints\model_epoch_31.pth `
    --features-dir data\stpls3d\img_features `
    --masks-dir data\stpls3d\img_class `
    --device cuda
```

Windows CMD:
```cmd
python training\evaluate.py ^
    --checkpoint checkpoints\model_epoch_31.pth ^
    --features-dir data\stpls3d\img_features ^
    --masks-dir data\stpls3d\img_class ^
    --device cuda
```

Prints per-class Precision, Recall, IoU, F1 and overall mIoU.

## 5. Use Trained Model

Copy the best checkpoint to `models/`:

Linux/macOS:
```bash
cp checkpoints/model_epoch_31.pth models/model_epoch_31_multiclass.pth
```

Windows PowerShell:
```powershell
Copy-Item checkpoints\model_epoch_31.pth models\model_epoch_31_multiclass.pth
```

Windows CMD:
```cmd
copy checkpoints\model_epoch_31.pth models\model_epoch_31_multiclass.pth
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
