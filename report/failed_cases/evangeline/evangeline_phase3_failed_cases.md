# Phase 3 failed cases: evangeline

Test set: 35 regions, **2 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Most unlike its true class (feature value, z-score vs training regions of that class) |
|---|---|---|---|---|---|
| p29_s74 | tumour | healthy | 627 | 101 | area 627 (z +3.5), laws_L5E5 316 (z -2.6), laws_L5S5 184 (z -2.3) |
| p29_s74 | healthy | tumour | 578 | 93 | area 578 (z +3.0), perimeter_area_ratio 0.247 (z -1.3), laws_L5E5 438 (z -1.3) |