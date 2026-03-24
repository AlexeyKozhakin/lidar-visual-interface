# Presentation Outline (Draft)

## Goal
Internal/supervisor update on paper progress, current results, and next actions.

## Target
- Duration: 10 minutes
- Slides: 12
- Audience: supervisor + research team

## Slide Plan
1. **Title**
   - Paper title (working)
   - Authors and affiliations
   - Date

2. **Problem & Motivation**
   - Why urban LiDAR semantic segmentation matters
   - Why practical GIS-ready outputs are needed

3. **Method Overview (3D->2D->3D)**
   - End-to-end pipeline blocks
   - Two output modes: building polygons + multiclass LAS

4. **Data**
   - STPLS3D: train/val/test split (230/50/42)
   - Malta scenes: St Paul's Bay, Xewkija, Gozo Rabat, Sliema
   - Note: Malta has no GT labels (qualitative only)

5. **Input Representation**
   - KNN-based encoding to 512x512 tensors
   - Channels: z_adj, n_z, n_r (+ RGB/class for artifacts)

6. **Training Setup**
   - U-Net + ResNet34
   - Optimizer/loss/epochs
   - Hardware: Tesla T4, CUDA 13.0

7. **Main Quantitative Results**
   - mIoU (multiclass): 0.2786
   - Building IoU: 0.8069
   - F1 / Precision / Recall: 0.8931 / 0.9019 / 0.8845

8. **Per-Class Results**
   - Per-class IoU/F1 table highlights
   - Best classes / weakest classes

9. **Baseline Context**
   - Baseline values used in manuscript
   - Explicit note on protocol mismatch (SensatUrban vs STPLS3D split)

10. **Qualitative Outputs**
   - Sample features and masks
   - Malta building extraction examples
   - 3D back-projection example

11. **Applications & Limitations**
   - GIS footprints, urban mapping workflows
   - Projection limits, generalization constraints

12. **Current Status & Next Week Plan**
   - Done vs pending items
   - Next actions: Malta runtime/metrics, figure completion, final text pass

## Asset Checklist
- [ ] Pipeline diagram (Figure 1)
- [ ] Dataset sample panel (train/val/test)
- [ ] Loss/mIoU curves
- [ ] Confusion matrix (val)
- [ ] Malta qualitative figures (3,4,5)
- [ ] Final baseline + per-class tables export

## Open Items Before Final Deck
- [ ] Runtime per tile/per scene
- [ ] Malta dataset metadata (CRS, sizes)
- [ ] Acknowledgements details (data provider, GlaDOS, funding)
- [ ] Final author order + emails
