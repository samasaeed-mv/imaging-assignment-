# Phase 3 failed cases: sama

Test set: 50 regions, **12 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p0_s61 | healthy | tumour | 36 | 49 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s62 | healthy | tumour | 68 | 89 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s62 | tumour | healthy | 12 | 99 | tumour region too small to measure shape/texture reliably |
| p0_s69 | healthy | tumour | 103 | 47 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s69 | healthy | tumour | 135 | 82 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s70 | healthy | tumour | 37 | 63 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s70 | healthy | tumour | 236 | 88 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s71 | healthy | tumour | 17 | 79 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s72 | healthy | tumour | 45 | 52 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s72 | healthy | tumour | 88 | 96 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s72 | healthy | tumour | 63 | 84 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p0_s72 | healthy | tumour | 158 | 104 | healthy dark structure (vessel or organ edge) that looks tumour-like |