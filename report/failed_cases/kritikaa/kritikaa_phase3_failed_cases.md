# Phase 3 failed cases: kritikaa

Test set: 17 regions, **3 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p22_s129 | tumour | healthy | 68 | 64 | tumour region too small to measure shape/texture reliably |
| p22_s129 | tumour | healthy | 39 | 28 | tumour region too small to measure shape/texture reliably |
| p22_s129 | tumour | healthy | 274 | 44 | tumour looks like normal liver (similar brightness/texture) |