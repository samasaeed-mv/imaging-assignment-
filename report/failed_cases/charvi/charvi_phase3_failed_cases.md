# Phase 3 failed cases: charvi

Test set: 27 regions, **8 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p1_s62 | healthy | tumour | 251 | 97 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s62 | healthy | tumour | 128 | 91 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s63 | tumour | healthy | 75 | 90 | tumour region too small to measure shape/texture reliably |
| p1_s64 | tumour | healthy | 47 | 92 | tumour region too small to measure shape/texture reliably |
| p1_s64 | healthy | tumour | 431 | 80 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s64 | healthy | tumour | 84 | 106 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s68 | healthy | tumour | 171 | 35 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s68 | healthy | tumour | 69 | 92 | healthy dark structure (vessel or organ edge) that looks tumour-like |