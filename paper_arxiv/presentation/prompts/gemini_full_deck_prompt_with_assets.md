# Gemini Slides Prompt (with local assets)

Create a 12-slide academic presentation in English.

Topic:
"From 3D LiDAR to 2D Segmentation and Back: A Practical Pipeline for Building Footprints and Multiclass Point-Cloud Labeling"

Audience:
Research supervisor and project team.

Style:
- Clean academic design.
- White background, dark text, blue accent.
- Minimal text per slide.
- Strong emphasis on figures and tables.

Use these local assets where relevant:
- `paper_arxiv/presentation/assets/figures/sf_3d_cloudcompare.png`
- `paper_arxiv/presentation/assets/figures/sf_test_img_rgb.png`
- `paper_arxiv/presentation/assets/figures/sf_test_img_features.png`
- `paper_arxiv/presentation/assets/figures/sf_test_img_class.png`
- `paper_arxiv/presentation/assets/figures/training_loss_curves.png`
- `paper_arxiv/presentation/assets/figures/training_miou_curves.png`
- `paper_arxiv/presentation/assets/figures/confusion_matrix_val.png`
- `paper_arxiv/presentation/assets/tables/main_metrics.csv`
- `paper_arxiv/presentation/assets/tables/per_class_metrics.csv`

Slide structure:
1) Title + authors + date
2) Problem and motivation
3) Pipeline overview (3D->2D->3D)
4) Datasets and sample artifacts
5) Feature encoding details
6) Training setup + hardware
7) Main quantitative results
8) Per-class performance highlights
9) Baseline context + comparability caveat
10) Qualitative outputs (current + placeholders for Malta)
11) Applications and limitations
12) Current status and next-week plan

Must include these facts:
- STPLS3D split: train 230, val 50, test 42.
- Multiclass mIoU: 0.2786.
- Building IoU: 0.8069.
- Binary F1: 0.8931.
- Precision: 0.9019.
- Recall: 0.8845.
- Hardware: NVIDIA Tesla T4 (CUDA 13.0), x86_64 CPU, 12.7 GB RAM.
- Malta scenes: St Paul's Bay, Xewkija, Gozo Rabat, Sliema.

Important caveat:
Baseline numbers and our numbers may follow different protocols/splits; comparisons are contextual, not direct SOTA claims.

Output requirements:
- Exactly 12 slides.
- Add 2-4 speaker-note bullets per slide.
- If an asset is missing, add a clear placeholder text box.
