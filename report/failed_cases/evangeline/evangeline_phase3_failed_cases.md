# Phase 3 failed cases: evangeline

Test set: 35 regions, **2 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Likely reason |
|---|---|---|---|---|---|
| p29_s74 | tumour | healthy | 627 | 101 | tumour looks like normal liver (similar brightness/texture) |
| p29_s74 | healthy | tumour | 578 | 93 | healthy dark structure (vessel or organ edge) that looks tumour-like |