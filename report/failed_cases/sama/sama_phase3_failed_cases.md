# Phase 3 failed cases: sama

Test set: 21 regions, **2 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p0_s67 | tumour | healthy | 73 | 98 | tumour region too small to measure shape/texture reliably |
| p0_s62 | tumour | healthy | 12 | 99 | tumour region too small to measure shape/texture reliably |