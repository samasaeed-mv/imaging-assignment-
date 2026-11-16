# Phase 3 failed cases: sama

Test set: 22 regions, **1 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p0_s68 | tumour | healthy | 30 | 100 | tumour region too small to measure shape/texture reliably |