# Phase 2 failed cases: evangeline (tumour segmentation)

Best method: **watershed** (mean Dice 0.600). Slices with Dice < 0.7: 10 of 22.

| Slice | Dice (watershed) | Ground-truth pixels | Predicted pixels | Likely reason |
|---|---|---|---|---|
| p29_s56 | 0.000 | 75 | 85 | very small tumour on this slice (75 pixels), near the top or bottom of the organ |
| p29_s77 | 0.044 | 80 | 1383 | very small tumour on this slice (80 pixels), near the top or bottom of the organ |
| p29_s76 | 0.139 | 172 | 1237 | very small tumour on this slice (172 pixels), near the top or bottom of the organ |
| p29_s57 | 0.288 | 178 | 266 | very small tumour on this slice (178 pixels), near the top or bottom of the organ |
| p29_s75 | 0.308 | 363 | 1252 | over-segmentation: the mask leaked into neighbouring tissue |
| p29_s58 | 0.468 | 261 | 337 | very small tumour on this slice (261 pixels), near the top or bottom of the organ |
| p29_s59 | 0.544 | 429 | 432 | boundary mismatch: right place, but the outline differs from the ground truth |
| p29_s74 | 0.607 | 745 | 841 | boundary mismatch: right place, but the outline differs from the ground truth |
| p29_s62 | 0.690 | 641 | 541 | boundary mismatch: right place, but the outline differs from the ground truth |
| p29_s61 | 0.694 | 560 | 449 | boundary mismatch: right place, but the outline differs from the ground truth |

Overlay images of the 3 worst slices are saved next to this file (red = predicted, green = ground truth).