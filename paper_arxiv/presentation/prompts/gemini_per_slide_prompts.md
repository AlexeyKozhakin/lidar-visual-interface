# Gemini Prompts — One Prompt Per Slide (Expanded Context + Unified Style)

## Global Style Instructions (apply to every slide)
Use these style rules on each slide:
- Visual style: professional academic + public-sector project tone.
- Palette: deep navy (#0B1F3A), teal (#0F766E), slate gray (#334155), light background (#F8FAFC), white cards.
- Typography: clear sans-serif, high readability, strong hierarchy.
- Layout: clean grid, generous margins, no clutter, no flashy effects.
- Charts/tables: simple, high contrast, labeled clearly.
- Iconography: minimal and consistent line icons only.
- Voice: factual, modest claims, reproducibility-oriented.
- Footer (small): "University of Malta | LiDAR Urban Segmentation Project".

Project context to reflect where relevant:
- This is an academic project at the University of Malta.
- Funding context: government-supported/publicly funded research project.
- Purpose: practical geospatial workflow for operational use, not only model novelty.

Core facts (use consistently when needed):
- Working title: "From 3D LiDAR to 2D Segmentation and Back: A Practical Pipeline for Building Footprints and Multiclass Point-Cloud Labeling"
- Authors:
  - Saviour Formosa (Associate Professor, University of Malta)
  - Tram Nguyen (PhD Student, University of Malta)
  - Dylan Seychell (Faculty Member, University of Malta)
  - Alexey Kozhakin (PhD Student, University of Malta)
- STPLS3D split: train 230, val 50, test 42 (total 322).
- Malta scenes: St Paul's Bay, Xewkija, Gozo Rabat, Sliema.
- Main metrics:
  - mIoU (multiclass): 0.2786
  - Building IoU: 0.8069
  - Binary F1: 0.8931
  - Precision: 0.9019
  - Recall: 0.8845
- Hardware: NVIDIA Tesla T4 (CUDA 13.0), x86_64 CPU, 12.7 GB RAM.
- Caveat: baseline context may use different protocols/splits; avoid direct SOTA claims.

---

## Slide 1 — Title
Create Slide 1 with a strong academic title layout.
Title: "From 3D LiDAR to 2D Segmentation and Back"
Subtitle: "A Practical Pipeline for Building Footprints and Multiclass Point-Cloud Labeling"
Add all four authors and affiliations (University of Malta).
Add date: March 2026.
Add one-line project context: "Academic, government-supported geospatial AI project."
Apply the global style instructions exactly.

## Slide 2 — Problem & Motivation
Create Slide 2 titled "Problem and Motivation".
Explain the operational gap:
- Urban LiDAR segmentation is compute-heavy.
- 3D methods are strong but often hard to deploy in GIS pipelines.
- Teams need outputs directly usable in planning and mapping workflows.
Add one takeaway box: "Goal: practical end-to-end pipeline from raw LAS to GIS-ready outputs."
Apply the global style instructions exactly.

## Slide 3 — Method Overview
Create Slide 3 titled "Method Overview (3D->2D->3D)".
Draw a clean left-to-right process diagram:
LAS input -> 250m tiling -> KNN feature encoding (512x512) -> U-Net segmentation -> tile stitching -> outputs.
Outputs block must show:
- Building footprints (Shapefile)
- Classified 3D point cloud (LAS)
Add a short note: "Design prioritizes robustness, throughput, and interoperability."
Apply the global style instructions exactly.

## Slide 4 — Datasets
Create Slide 4 titled "Datasets and Evaluation Scope".
Two-column layout:
Left (STPLS3D): split 230/50/42, multiclass supervised training/validation.
Right (Malta): 4 scenes (St Paul's Bay, Xewkija, Gozo Rabat, Sliema), qualitative inference only.
Add clear caveat: "No Malta ground truth currently available for quantitative evaluation."
Apply the global style instructions exactly.

## Slide 5 — Dataset Artifacts Example
Create Slide 5 titled "Dataset Artifacts Example (SanFrancisco_500_500)".
Insert and align these 4 assets in one row:
- paper_arxiv/presentation/assets/figures/sf_3d_cloudcompare.png
- paper_arxiv/presentation/assets/figures/sf_test_img_rgb.png
- paper_arxiv/presentation/assets/figures/sf_test_img_features.png
- paper_arxiv/presentation/assets/figures/sf_test_img_class.png
Label each panel clearly.
Caption: "CloudCompare 3D view and aligned 2D artifacts used in training workflow."
Apply the global style instructions exactly.

## Slide 6 — Training Setup
Create Slide 6 titled "Training Setup and Compute Environment".
Include concise technical bullets:
- Model: U-Net + ResNet34 encoder
- Training objective: multiclass + binary segmentation tracks
- Optimizer/loss: Adam + Dice variants
- Hardware: Tesla T4, CUDA 13.0, x86_64 CPU, 12.7 GB RAM
Add one sentence: "Configuration emphasizes reproducibility and practical deployment constraints."
Apply the global style instructions exactly.

## Slide 7 — Main Quantitative Results
Create Slide 7 titled "Main Quantitative Results".
Build a compact results card/table with:
- mIoU (multiclass): 0.2786
- Building IoU: 0.8069
- Binary F1: 0.8931
- Precision: 0.9019
- Recall: 0.8845
Add note at bottom: "Runtime per tile/scene will be finalized after dedicated timing runs."
Apply the global style instructions exactly.

## Slide 8 — Per-Class Performance
Create Slide 8 titled "Per-Class Performance".
Use `paper_arxiv/presentation/assets/tables/per_class_metrics.csv`.
Show Top-5 and Bottom-5 classes by IoU in a clean comparison layout.
Add interpretation bullets:
- Strong performance on dominant classes (e.g., Building).
- Lower performance on rare/complex classes.
- Class imbalance likely contributes to spread.
Apply the global style instructions exactly.

## Slide 9 — Training Diagnostics
Create Slide 9 titled "Training Diagnostics".
Insert:
- paper_arxiv/presentation/assets/figures/training_loss_curves.png
- paper_arxiv/presentation/assets/figures/training_miou_curves.png
- paper_arxiv/presentation/assets/figures/confusion_matrix_val.png
Add 2-3 concise observations on convergence and validation behavior.
Avoid overclaiming; keep statements evidence-based.
Apply the global style instructions exactly.

## Slide 10 — Baseline Context
Create Slide 10 titled "Baseline Context and Fair Comparison".
Add a small table with selected baseline context values and our values.
Include a prominent caution box:
"Comparisons are contextual; protocols/splits differ, so this is not a direct SOTA claim."
Add one bullet: "Primary contribution is pipeline integration from LAS to GIS-ready outputs."
Apply the global style instructions exactly.

## Slide 11 — Applications & Limitations
Create Slide 11 titled "Applications and Limitations".
Two balanced columns:
Applications:
- Building footprint extraction for GIS
- Urban planning/infrastructure mapping
- Practical integration with existing geospatial tools
Limitations:
- 3D->2D projection can lose vertical detail
- Domain shift to unseen regions/sensors
- Runtime and Malta quantitative validation still pending
Apply the global style instructions exactly.

## Slide 12 — Status & Next Steps
Create Slide 12 titled "Current Status and Next Steps".
Left block "Completed":
- Training completed
- Core metrics integrated
- Literature base expanded to 28 references
Right block "Next Week Plan":
- Add Malta quantitative/runtime results
- Finalize missing figures and acknowledgements
- Continue manuscript refinement toward final review-ready draft
Final line: "Target: submission-ready draft after supervisor/team review."
Apply the global style instructions exactly.
