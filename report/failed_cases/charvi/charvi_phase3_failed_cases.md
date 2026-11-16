# Phase 3 failed cases: charvi

Test set: 44 regions, **5 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Most unlike its true class (feature value, z-score vs training regions of that class) |
|---|---|---|---|---|---|
| p1_s57 | tumour | healthy | 65 | 90 | lbp_4 0.246 (z -1.6), glcm_homogeneity 0.864 (z +1.6), glcm_entropy 1.72 (z -1.6) |
| p1_s60 | tumour | healthy | 250 | 67 | lbp_0 0.008 (z +8000000.0), lbp_9 0.064 (z +38.4), lbp_1 0.02 (z +25.1) |
| p1_s61 | tumour | healthy | 50 | 68 | lbp_9 0.06 (z +35.9), glcm_homogeneity 1 (z +4.8), glcm_energy 1 (z +4.2) |
| p1_s68 | tumour | healthy | 56 | 76 | glcm_energy 0.698 (z +2.0), lbp_6 0.214 (z +2.0), glcm_entropy 1.37 (z -2.0) |
| p1_s69 | healthy | tumour | 338 | 74 | perimeter_area_ratio 0.232 (z -1.7), area 338 (z +1.1), glcm_energy 0.368 (z -0.9) |