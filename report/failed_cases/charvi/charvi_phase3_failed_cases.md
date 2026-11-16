# Phase 3 failed cases: charvi

Test set: 44 regions, **5 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p1_s57 | tumour | healthy | 65 | 90 | tumour region too small to measure shape/texture reliably |
| p1_s60 | tumour | healthy | 250 | 67 | tumour looks like normal liver (similar brightness/texture) |
| p1_s61 | tumour | healthy | 50 | 68 | tumour region too small to measure shape/texture reliably |
| p1_s68 | tumour | healthy | 56 | 76 | tumour region too small to measure shape/texture reliably |
| p1_s69 | healthy | tumour | 338 | 74 | healthy dark structure (vessel or organ edge) that looks tumour-like |