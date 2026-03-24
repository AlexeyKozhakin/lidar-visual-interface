# Gemini Slides Prompt — Key Slides

## Prompt A: Method Slide
Create one slide titled "Method Overview (3D->2D->3D)".
Include a left-to-right pipeline diagram with these blocks:
LAS input -> 250m tiling -> KNN feature encoding (512x512) -> U-Net segmentation -> stitching -> outputs (Shapefile polygons / classified LAS).
Use short labels only and one short takeaway sentence.

## Prompt B: Results Slide
Create one slide titled "Main Quantitative Results".
Include a compact table with these values:
- mIoU (multiclass): 0.2786
- Building IoU: 0.8069
- Binary F1: 0.8931
- Precision: 0.9019
- Recall: 0.8845
Add a one-line note: "Runtime values to be finalized after additional timing runs."

## Prompt C: Baseline Caveat Slide
Create one slide titled "Baseline Context and Fair Comparison".
Include:
- 3 bullets explaining protocol mismatch risks.
- One warning box: "Comparisons are contextual, not direct SOTA claims."
- Optional small table with baseline mIoU and Building IoU values.

## Prompt D: Status Slide
Create one slide titled "Current Status and Next Week Plan".
Split into two columns:
Left: completed tasks (training, metrics extraction, 28 references, draft sections).
Right: next-week tasks (Malta calculations, final figures, presentation finalization, text refinement).
Add a final line: "Target: submission-ready draft after team review."
