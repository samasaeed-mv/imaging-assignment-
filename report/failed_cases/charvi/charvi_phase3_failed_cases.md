# Phase 3 failed cases: charvi

Test set: 26 regions, **7 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p1_s64 | tumour | healthy | 47 | 92 | tumour region too small to measure shape/texture reliably |
| p1_s65 | healthy | tumour | 291 | 92 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s70 | healthy | tumour | 50 | 79 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s70 | healthy | tumour | 60 | 35 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s72 | healthy | tumour | 23 | 63 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s72 | healthy | tumour | 25 | 65 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p1_s72 | healthy | tumour | 589 | 54 | healthy dark structure (vessel or organ edge) that looks tumour-like |