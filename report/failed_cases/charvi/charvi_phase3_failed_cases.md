# Phase 3 failed cases: charvi

Test set: 27 regions, **2 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p1_s57 | tumour | healthy | 65 | 89 | tumour region too small to measure shape/texture reliably |
| p1_s60 | tumour | healthy | 37 | 82 | tumour region too small to measure shape/texture reliably |