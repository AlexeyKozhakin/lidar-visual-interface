# Polygon Pipeline Plan

## Goal

Add a first-class polygon extraction workflow to the package pipeline so that
the package can produce either:

1. classified LAS output from the multiclass model, or
2. building polygons / shapefiles from the binary building model.

The implementation must preserve the existing classification workflow and avoid
mixing binary and multiclass logic in an ad hoc way.


## Current State

The repository already contains the required building blocks:

1. Multiclass package pipeline:
   - `musac_las_classifier/las_classification_pipeline.py`
2. Binary building predictor:
   - `musac_las_classifier/predictor_building_segmentation/predict_building_segmentation.py`
3. Polygon generation from predicted masks:
   - `musac_las_classifier/polygon_generator/polygon_generator.py`
4. Older end-to-end polygon script:
   - `main_poligon.py`

What is missing is a clean package-level orchestration that exposes polygon
generation as a supported workflow in the same way classified LAS export is
already exposed.


## Design Constraints

1. Do not break the existing multiclass LAS classification API.
2. Keep binary-building and multiclass workflows explicit.
3. Remove hardcoded model paths from binary prediction code.
4. Make package imports self-contained; avoid depending on top-level config
   modules from inside package functions.
5. Keep outputs organized in dedicated work directories.
6. Polygon generation must run on the joined 1 km x 1 km prediction mask, not
   on individual 250 m x 250 m tiles.
7. CLI support is not part of the current implementation phase.


## Target API

### Config additions

Extend `LasPipelineConfig` with polygon-specific fields:

1. `building_checkpoint_path: Optional[str]`
2. `building_encoder_weights_path: Optional[str]`
3. `min_polygon_area: int`
4. `contour_thickness: int`

These fields are for the binary building workflow only.

The config object should remain unified for both workflows. To keep usage
explicit and clean, add constructor helpers such as:

1. `LasPipelineConfig.for_multiclass(...)`
2. `LasPipelineConfig.for_polygon_extraction(...)`


### Pipeline methods

Keep the current multiclass workflow and add a separate polygon workflow.

Existing multiclass path:

1. `slice_las()`
2. `transform_to_tensor()`
3. `prepare_features()`
4. `predict()`
5. `postprocess()`
6. `export_las(output_las_path)`

New polygon path:

1. `predict_buildings()`
2. `postprocess_buildings()`
3. `export_polygons(output_shp_dir)`
4. `run_polygon_extraction(output_shp_dir)`

The polygon workflow should reuse common preprocessing stages but use a
different prediction branch and different output directories.

The existing multiclass methods and their behavior must remain backward
compatible.


## Required Refactors

### 1. Binary predictor cleanup

File:
- `musac_las_classifier/predictor_building_segmentation/predict_building_segmentation.py`

Required changes:

1. Accept checkpoint path as argument.
2. Accept encoder weights path as argument.
3. Remove hardcoded local model path.
4. Match the style of the multiclass predictor entry point.


### 2. Polygon generator cleanup

File:
- `musac_las_classifier/polygon_generator/polygon_generator.py`

Required changes:

1. Use package-safe imports.
2. Remove dependency on external runtime config inside callable functions.
3. Keep `main_polygon_generator(...)` fully parameter-driven.


### 3. Pipeline orchestration

File:
- `musac_las_classifier/las_classification_pipeline.py`

Required changes:

1. Add polygon-specific working directories:
   - `img_predict_building`
   - `img_predict_building_join`
   - `img_contours`
   - `polygons_shp`
2. Add building prediction method.
3. Add polygon export method.
4. Add high-level polygon workflow runner.
5. Keep multiclass workflow behavior unchanged.


### 4. Package exports

File:
- `musac_las_classifier/__init__.py`

Possible changes:

1. Export any new config helpers if needed.
2. Ensure public API remains simple and discoverable.


### 5. Package data

Files:
- `pyproject.toml`

Required changes:

1. Ensure binary building model weights are included in package data.
2. Keep multiclass model packaging unchanged.
3. Make installed package self-contained for both workflows.


### 6. CLI support

CLI support is intentionally deferred. The current target is package-level
class methods only.


## Proposed Output Layout

Inside the package workdir:

1. Shared preprocessing outputs:
   - `las_cut/`
   - `tensor/`
   - `img_features/`
2. Multiclass outputs:
   - `img_predict_multiclass/`
   - `img_predict_multiclass_join/`
3. Polygon outputs:
   - `img_predict_building/`
   - `img_predict_building_join/`
   - `img_contours/`
   - `polygons_shp/`


## Execution Plan

### Phase 1

Refactor binary building prediction code so it is parameter-driven and package-safe.

### Phase 2

Refactor polygon generation code so it is package-safe and does not depend on
top-level config modules.

### Phase 3

Integrate polygon workflow into `LasClassificationPipeline` with dedicated
methods and work directories.

### Phase 4

Add CLI support and documentation examples.

### Phase 5

Run end-to-end verification:

1. multiclass LAS workflow still works,
2. binary polygon workflow runs end-to-end,
3. shapefiles are written correctly,
4. output directory structure is stable.


## Validation Checklist

Before calling the work complete:

1. `run()` still produces joined multiclass masks.
2. `export_las()` still exports classified LAS.
3. `run_polygon_extraction()` produces:
   - joined binary prediction images,
   - contour preview images,
   - `.shp/.shx/.dbf` files.
4. No hardcoded local model paths remain in polygon workflow code.
5. Package imports work after `pip install -e .`.


## Open Questions

These points are now fixed for implementation:

1. Polygon export is generated only from the joined scene-level prediction
   image after tile stitching.
2. `min_polygon_area` and `contour_thickness` are part of the package pipeline
   configuration.
3. Binary and multiclass settings live in one config object, with explicit
   helper constructors for each workflow.
4. Existing multiclass pipeline behavior must remain unchanged.
