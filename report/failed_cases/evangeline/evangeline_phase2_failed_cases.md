# Phase 2 failed cases: evangeline (tumour segmentation)

Best method: **sauvola** (mean Dice 0.638). Slices with Dice < 0.7: 8 of 22.

| Slice | Dice (sauvola) | Ground-truth pixels | Predicted pixels | Likely reason |
|---|---|---|---|---|
| p29_s56 | 0.000 | 75 | 0 | missed completely: nothing was segmented |
| p29_s77 | 0.202 | 80 | 256 | very small tumour on this slice (80 pixels), near the top or bottom of the organ |
| p29_s76 | 0.333 | 172 | 416 | very small tumour on this slice (172 pixels), near the top or bottom of the organ |
| p29_s57 | 0.368 | 178 | 192 | very small tumour on this slice (178 pixels), near the top or bottom of the organ |
| p29_s58 | 0.505 | 261 | 294 | very small tumour on this slice (261 pixels), near the top or bottom of the organ |
| p29_s74 | 0.556 | 745 | 525 | boundary mismatch: right place, but the outline differs from the ground truth |
| p29_s75 | 0.561 | 363 | 357 | boundary mismatch: right place, but the outline differs from the ground truth |
| p29_s59 | 0.672 | 429 | 279 | boundary mismatch: right place, but the outline differs from the ground truth |

Overlay images of the 3 worst slices are saved next to this file (red = predicted, green = ground truth).