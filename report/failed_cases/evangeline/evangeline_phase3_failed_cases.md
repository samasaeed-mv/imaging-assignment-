# Phase 3 failed cases: evangeline

Test set: 36 regions, **7 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p29_s57 | healthy | tumour | 62 | 74 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p29_s58 | tumour | healthy | 56 | 89 | tumour region too small to measure shape/texture reliably |
| p29_s62 | tumour | healthy | 166 | 93 | tumour looks like normal liver (similar brightness/texture) |
| p29_s63 | tumour | healthy | 233 | 88 | tumour looks like normal liver (similar brightness/texture) |
| p29_s63 | healthy | tumour | 209 | 103 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p29_s63 | healthy | tumour | 94 | 41 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p29_s63 | healthy | tumour | 241 | 86 | healthy dark structure (vessel or organ edge) that looks tumour-like |