# Results TODO Registry

This file is the contract for all numeric values that will be inserted later.
Use exact placeholder tokens in manuscript sections and update status as values become available.

| Placeholder | Meaning | Where Used | Source Script/Notebook | Status | Notes |
|---|---|---|---|---|---|
| [[METRIC_STPLS3D_MIOU_MULTICLASS]] | mIoU on STPLS3D for multiclass model | Results, Abstract, Conclusion | TODO | pending | |
| [[METRIC_STPLS3D_IOU_BUILDING]] | IoU for building class (or binary building IoU) | Results, Abstract | TODO | pending | |
| [[METRIC_STPLS3D_F1_BINARY]] | F1 for binary building segmentation | Results, Abstract | TODO | pending | |
| [[METRIC_STPLS3D_PRECISION_BINARY]] | Precision for binary building segmentation | Results | TODO | pending | |
| [[METRIC_STPLS3D_RECALL_BINARY]] | Recall for binary building segmentation | Results | TODO | pending | |
| [[METRIC_STPLS3D_PER_CLASS_MIOU]] | Per-class IoU/mIoU summary | Results | TODO | pending | define table format |
| [[TIME_INFERENCE_PER_TILE_SEC]] | Inference time per tile (seconds) | Experiments, Results | TODO | pending | report hardware |
| [[TIME_PIPELINE_PER_TILE_SEC]] | End-to-end processing time per tile (seconds) | Experiments, Results | TODO | pending | include preprocessing+postprocessing |
| [[TIME_PER_SCENE_MIN]] | End-to-end processing time per scene (minutes) | Results | TODO | pending | optional |
| [[HW_CPU_MODEL]] | CPU model used in benchmarks | Experiments | TODO | pending | |
| [[HW_GPU_MODEL]] | GPU model used in benchmarks | Experiments | TODO | pending | if used |
| [[HW_RAM_GB]] | RAM used in benchmarks | Experiments | TODO | pending | |
| [[DATA_STPLS3D_SPLIT_DESC]] | Train/val/test split description | Datasets subsection | TODO | pending | scene/tile based |
| [[DATA_MALTA_DESC]] | Malta data description (areas, volume) | Datasets subsection | TODO | pending | source attribution |
| [[ACK_MALTA_DATA_PROVIDER]] | Who provided Malta dataset | Acknowledgements | TODO | pending | exact institution/person |
| [[ACK_GLADOS_RESOURCE]] | GlaDOS allocation details | Acknowledgements | TODO | pending | project/allocation id if any |

## Replacement rule
- Keep placeholders in double square brackets exactly as written.
- Replace only after values are validated.
- Update `Status` to `done` after replacement.
