# Speaker Notes (10-minute version)

## Slide 1 — Title (0:30)
This talk summarizes progress on our LiDAR paper and the next steps to finalize it.

## Slide 2 — Motivation (0:45)
Urban LiDAR segmentation is useful but often hard to operationalize. Our focus is a practical pipeline that produces GIS-ready outputs.

## Slide 3 — Method (1:00)
The pipeline converts 3D point clouds to 2D feature images, runs segmentation with U-Net, and projects predictions back to 3D.

## Slide 4 — Data (0:50)
Training/validation/testing use STPLS3D with prepared split 230/50/42. Malta is used for cross-domain qualitative inference.

## Slide 5 — Input Encoding (0:50)
KNN aggregation creates compact features encoding terrain-corrected elevation and geometric proxies.

## Slide 6 — Training Setup (0:45)
U-Net ResNet34, Dice loss, Adam. Training executed on Tesla T4.

## Slide 7 — Main Results (1:00)
Report mIoU 0.2786 and building-focused metrics (IoU/F1/Prec/Rec). Emphasize which values are already final and which remain pending.

## Slide 8 — Per-Class (0:50)
Show performance spread by class. Mention class imbalance and hard classes.

## Slide 9 — Baselines (0:45)
Show baseline context and clearly note protocol differences to avoid overclaiming.

## Slide 10 — Qualitative Malta (1:00)
Present example outputs and discuss practical usability.

## Slide 11 — Applications and Limits (0:45)
Applications: GIS footprints and mapping. Limits: projection information loss and domain shift.

## Slide 12 — Status and Plan (1:00)
State done items, next-week tasks (Malta calculations + deck finalization), and request feedback.
