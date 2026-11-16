# Phase 3 failed cases: sama

Test set: 50 regions, **12 misclassified**.

| Slice | True | Predicted | Area (px) | Mean HU | Most unlike its true class (feature value, z-score vs training regions of that class) |
|---|---|---|---|---|---|
| p0_s61 | healthy | tumour | 36 | 49 | mean_HU 48.6 (z -2.0), skewness 0.676 (z +2.0), glcm_homogeneity 0.656 (z -1.7) |
| p0_s62 | healthy | tumour | 68 | 89 | hu_2 4.58 (z +3.3), eccentricity 0.24 (z -3.2), hu_6 -8.57 (z -1.8) |
| p0_s62 | tumour | healthy | 12 | 99 | skewness 0.955 (z +2.1), kurtosis 1.18 (z +2.1), circularity 1.29 (z +2.0) |
| p0_s69 | healthy | tumour | 103 | 47 | mean_HU 46.7 (z -2.1), glcm_correlation 0.897 (z +1.1), hu_7 -7.99 (z -1.1) |
| p0_s69 | healthy | tumour | 135 | 82 | hu_5 -7.03 (z -1.2), hu_6 -4.94 (z -1.1), hu_7 -7.44 (z -1.0) |
| p0_s70 | healthy | tumour | 37 | 63 | mean_HU 62.9 (z -1.5), hu_5 -8.63 (z -1.4), hu_6 -5.81 (z -1.3) |
| p0_s70 | healthy | tumour | 236 | 88 | eccentricity 0.474 (z -1.8), hu_2 3.21 (z +1.6), hu_5 -8.34 (z -1.4) |
| p0_s71 | healthy | tumour | 17 | 79 | hu_5 -10.2 (z -1.6), kurtosis -1.25 (z -1.5), hu_6 -6.63 (z -1.4) |
| p0_s72 | healthy | tumour | 45 | 52 | mean_HU 52 (z -1.9), hu_6 6.08 (z +1.1), glcm_correlation 0.875 (z +1.0) |
| p0_s72 | healthy | tumour | 88 | 96 | hu_5 -9.04 (z -1.5), hu_6 -5.99 (z -1.3), hu_7 8.78 (z +1.0) |
| p0_s72 | healthy | tumour | 63 | 84 | glcm_correlation 0.926 (z +1.2), kurtosis -0.802 (z -0.8), eccentricity 0.902 (z +0.7) |
| p0_s72 | healthy | tumour | 158 | 104 | hu_7 -6.8 (z -0.9), hu_6 5.17 (z +0.9), perimeter_area_ratio 0.366 (z -0.8) |