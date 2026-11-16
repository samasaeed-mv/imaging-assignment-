# Phase 3 failed cases: kritikaa

Test set: 17 regions, **3 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Most unlike its true class (feature value, z-score vs training regions of that class) |
|---|---|---|---|---|---|
| p22_s129 | tumour | healthy | 68 | 64 | glcm_correlation 0.268 (z -9.0), perimeter_area_ratio 0.426 (z +6.7), glcm_contrast 2.04 (z +4.9) |
| p22_s129 | tumour | healthy | 39 | 28 | perimeter_area_ratio 0.538 (z +9.4), variance_HU 202 (z -4.6), circularity 1.11 (z +4.1) |
| p22_s129 | tumour | healthy | 274 | 44 | glcm_homogeneity 0.779 (z +2.8), perimeter_area_ratio 0.26 (z +2.7), solidity 0.87 (z -2.7) |