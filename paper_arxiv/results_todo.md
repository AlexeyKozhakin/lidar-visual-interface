# Results TODO Registry

This file is the contract for all numeric values that will be inserted later.
Use exact placeholder tokens in manuscript sections and update status as values become available.

| Placeholder | Meaning | Where Used | Source Script/Notebook | Status | Notes |
|---|---|---|---|---|---|
| [[METRIC_STPLS3D_MIOU_MULTICLASS]] | mIoU on STPLS3D for multiclass model | Results, Abstract, Conclusion | `paper_arxiv/result/raw/epoch_summary.csv` | done | Best val mIoU (epoch 45): `0.2786` |
| [[METRIC_STPLS3D_IOU_BUILDING]] | IoU for building class (or binary building IoU) | Results, Abstract | `paper_arxiv/result/raw/metrics_log.csv` | done | Val class=1 IoU at best epoch: `0.8069` |
| [[METRIC_STPLS3D_F1_BINARY]] | F1 for binary building segmentation | Results, Abstract | `paper_arxiv/result/raw/metrics_log.csv` | done | Val class=1 F1 at best epoch: `0.8931` |
| [[METRIC_STPLS3D_PRECISION_BINARY]] | Precision for binary building segmentation | Results | `paper_arxiv/result/raw/metrics_log.csv` | done | Val class=1 precision at best epoch: `0.9019` |
| [[METRIC_STPLS3D_RECALL_BINARY]] | Recall for binary building segmentation | Results | `paper_arxiv/result/raw/metrics_log.csv` | done | Val class=1 recall at best epoch: `0.8845` |
| [[METRIC_STPLS3D_PER_CLASS_MIOU]] | Per-class IoU/mIoU summary | Results | `paper_arxiv/result/raw/metrics_log.csv` | done | Per-class IoU/F1 table inserted in manuscript (Val, epoch 45) |
| [[TIME_INFERENCE_PER_TILE_SEC]] | Inference time per tile (seconds) | Experiments, Results | TODO | pending | report hardware |
| [[TIME_PIPELINE_PER_TILE_SEC]] | End-to-end processing time per tile (seconds) | Experiments, Results | TODO | pending | include preprocessing+postprocessing |
| [[TIME_PER_SCENE_MIN]] | End-to-end processing time per scene (minutes) | Results | TODO | pending | optional |
| [[HW_CPU_MODEL]] | CPU model used in benchmarks | Experiments | `paper_arxiv/result/raw/hardware_info.json` | done | `x86_64` |
| [[HW_GPU_MODEL]] | GPU model used in benchmarks | Experiments | user-provided `nvidia-smi` snapshot | done | `NVIDIA Tesla T4 (CUDA 13.0, Driver 580.82.07)` |
| [[HW_RAM_GB]] | RAM used in benchmarks | Experiments | `paper_arxiv/result/raw/hardware_info.json` | done | `12.7` |
| [[DATA_STPLS3D_SPLIT_DESC]] | Train/val/test split description | Datasets subsection | user-provided split counts | done | Train=230, Val=50, Test=42, Total=322 |
| [[DATA_MALTA_DESC]] | Malta data description (areas, volume) | Datasets subsection | TODO | pending | source attribution |
| [[ACK_MALTA_DATA_PROVIDER]] | Who provided Malta dataset | Acknowledgements | TODO | pending | exact institution/person |
| [[ACK_GLADOS_RESOURCE]] | GlaDOS allocation details | Acknowledgements | TODO | pending | project/allocation id if any |

## Replacement rule
- Keep placeholders in double square brackets exactly as written.
- Replace only after values are validated.
- Update `Status` to `done` after replacement.
