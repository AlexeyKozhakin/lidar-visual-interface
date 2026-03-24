# Gemini Slides Prompt — Full Deck

Create a 12-slide academic presentation in English.

Topic:
"From 3D LiDAR to 2D Segmentation and Back: A Practical Pipeline for Building Footprints and Multiclass Point-Cloud Labeling"

Audience:
Research supervisor and project team.

Style requirements:
- Academic and clear, minimal text per slide.
- Clean white background, dark text, blue accent.
- No decorative elements.
- Use data-focused layouts and readable tables.
- Add speaker notes per slide (2-4 bullet points).

Slide structure:
1) Title + authors + affiliation
2) Problem and motivation
3) Pipeline overview (3D->2D->3D)
4) Datasets (STPLS3D + Malta)
5) Feature encoding details
6) Training setup and hardware
7) Main quantitative results
8) Per-class IoU/F1 highlights
9) Baseline context and comparability caveat
10) Qualitative Malta examples
11) Applications and limitations
12) Current status + next-week plan

Facts to include:
- STPLS3D split: train 230, val 50, test 42.
- Multiclass mIoU: 0.2786.
- Building IoU: 0.8069.
- Binary F1: 0.8931; Precision: 0.9019; Recall: 0.8845.
- Hardware: NVIDIA Tesla T4 (CUDA 13.0), x86_64 CPU, 12.7 GB RAM.
- Malta scenes: St Paul's Bay, Xewkija, Gozo Rabat, Sliema.

Important caveat slide:
Mention that baseline numbers and our numbers may follow different protocols/splits; comparison is contextual, not a strict benchmark claim.

Output expectations:
- 12 slides exactly.
- Suggested figure placeholders where images should be inserted.
- One final slide with action items for next week.
