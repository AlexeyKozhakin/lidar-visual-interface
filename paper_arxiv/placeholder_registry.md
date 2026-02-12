# Placeholder Registry

## Our model metrics (from evaluation)

- [[METRIC_STPLS3D_MIOU_MULTICLASS]] — mIoU on STPLS3D for multiclass model
- [[METRIC_STPLS3D_IOU_BUILDING]] — IoU for building class
- [[METRIC_STPLS3D_F1_BINARY]] — F1 for binary building segmentation
- [[METRIC_STPLS3D_PRECISION_BINARY]] — Precision for binary building segmentation
- [[METRIC_STPLS3D_RECALL_BINARY]] — Recall for binary building segmentation
- [[METRIC_STPLS3D_PER_CLASS_MIOU]] — Per-class IoU table (multiclass model)

## Baseline metrics (from published papers)

- [[BL_POINTNET_MIOU]] — PointNet mIoU on STPLS3D (from Hu et al. BMVC 2022)
- [[BL_POINTNETPP_MIOU]] — PointNet++ mIoU on STPLS3D
- [[BL_DGCNN_MIOU]] — DGCNN mIoU on STPLS3D

## Timing

- [[TIME_INFERENCE_PER_TILE_SEC]] — Inference time per tile (seconds)
- [[TIME_PIPELINE_PER_TILE_SEC]] — End-to-end processing time per tile (seconds)
- [[TIME_PER_SCENE_MIN]] — End-to-end time per scene (minutes)

## Hardware

- [[HW_CPU_MODEL]] — CPU model used in benchmarks
- [[HW_GPU_MODEL]] — GPU model used in benchmarks
- [[HW_RAM_GB]] — RAM in GB

## Dataset

- [[DATA_STPLS3D_SPLIT_DESC]] — Train/val/test split description (number of tiles)
- [[DATA_MALTA_DESC]] — Malta data description (areas, volume, coordinate system)

## Acknowledgements

- [[ACK_MALTA_DATA_PROVIDER]] — Who provided Malta dataset
- [[ACK_GLADOS_RESOURCE]] — GlaDOS allocation details

## Notes

- All placeholders appear in `main.tex` using `\placeholder{TOKEN}` command
- Replace only after values are validated
- Baseline numbers should be extracted from: Chen & Hu et al., "STPLS3D", BMVC 2022 (Table in Section 5)
- Known approximate ranges from web: KPConv ~70% mIoU (SyntheticV3), MinkowskiNet ~46.5% mIoU (WMSC real)
