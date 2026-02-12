# 5. Results (Template)

## 5.1 Quantitative Results

### Main Benchmark Table (STPLS3D)

| Model | mIoU | Building IoU | F1 (Binary) | Precision (Binary) | Recall (Binary) |
|---|---:|---:|---:|---:|---:|
| Ours (Multiclass U-Net ResNet34) | [[METRIC_STPLS3D_MIOU_MULTICLASS]] | [[METRIC_STPLS3D_IOU_BUILDING]] | [[METRIC_STPLS3D_F1_BINARY]] | [[METRIC_STPLS3D_PRECISION_BINARY]] | [[METRIC_STPLS3D_RECALL_BINARY]] |

### Runtime Table

| Scenario | Inference / tile (s) | End-to-end / tile (s) | End-to-end / scene (min) | Hardware |
|---|---:|---:|---:|---|
| Ours | [[TIME_INFERENCE_PER_TILE_SEC]] | [[TIME_PIPELINE_PER_TILE_SEC]] | [[TIME_PER_SCENE_MIN]] | CPU: [[HW_CPU_MODEL]], GPU: [[HW_GPU_MODEL]], RAM: [[HW_RAM_GB]] |

### Per-class Summary (Multiclass)

- [[METRIC_STPLS3D_PER_CLASS_MIOU]]
- TODO: insert per-class table or compact top/bottom classes summary.

## 5.2 Qualitative Results

- Figure TODO: feature maps (St Paul's Bay, Xewkija, Gozo Rabat).
- Figure TODO: building segmentation outputs.
- Figure TODO: multiclass segmentation outputs.
- Figure TODO: 3D colored/classified LAS outputs (e.g., Sliema).
- TODO: explain typical failure cases and edge conditions.

## Notes
- Keep this file as the canonical source for result tables during drafting.
- After final numbers are available, copy finalized tables into the main manuscript.
