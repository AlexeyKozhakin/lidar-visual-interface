# Placeholder Registry

## Detected existing placeholders/tokens

From `paper_arxiv/outline.md`:
- METRICS_TBD

From `paper_arxiv/sections/related_work.md`:
- no numeric placeholders yet

## Standard placeholders to use from now on

- [[METRIC_STPLS3D_MIOU_MULTICLASS]]
- [[METRIC_STPLS3D_IOU_BUILDING]]
- [[METRIC_STPLS3D_F1_BINARY]]
- [[METRIC_STPLS3D_PRECISION_BINARY]]
- [[METRIC_STPLS3D_RECALL_BINARY]]
- [[METRIC_STPLS3D_PER_CLASS_MIOU]]
- [[TIME_INFERENCE_PER_TILE_SEC]]
- [[TIME_PIPELINE_PER_TILE_SEC]]
- [[TIME_PER_SCENE_MIN]]
- [[HW_CPU_MODEL]]
- [[HW_GPU_MODEL]]
- [[HW_RAM_GB]]
- [[DATA_STPLS3D_SPLIT_DESC]]
- [[DATA_MALTA_DESC]]
- [[ACK_MALTA_DATA_PROVIDER]]
- [[ACK_GLADOS_RESOURCE]]

## Migration note

- Replace legacy token `METRICS_TBD` with specific placeholders from this registry.
