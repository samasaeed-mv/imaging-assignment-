# Phase 3 failed cases: evangeline

Test set: 37 regions, **5 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p29_s62 | tumour | healthy | 166 | 93 | tumour looks like normal liver (similar brightness/texture) |
| p29_s70 | tumour | healthy | 213 | 97 | tumour looks like normal liver (similar brightness/texture) |
| p29_s63 | healthy | tumour | 241 | 86 | healthy dark structure (vessel or organ edge) that looks tumour-like |
| p29_s71 | tumour | healthy | 281 | 102 | tumour looks like normal liver (similar brightness/texture) |
| p29_s72 | tumour | healthy | 337 | 104 | tumour looks like normal liver (similar brightness/texture) |