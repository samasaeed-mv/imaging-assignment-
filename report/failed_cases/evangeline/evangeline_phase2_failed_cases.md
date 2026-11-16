# Phase 2 failed cases: evangeline (tumour segmentation)

Best method: **chanvese** (mean Dice 0.608). Slices with Dice < 0.7: 10 of 22.

| Slice | Dice (chanvese) | Ground-truth pixels | Predicted pixels | Likely reason |
|---|---|---|---|---|
| p29_s56 | 0.000 | 75 | 0 | missed completely: nothing was segmented |
| p29_s77 | 0.075 | 80 | 1061 | very small tumour on this slice (80 pixels), near the top or bottom of the organ |
| p29_s76 | 0.218 | 172 | 938 | very small tumour on this slice (172 pixels), near the top or bottom of the organ |
| p29_s57 | 0.382 | 178 | 178 | very small tumour on this slice (178 pixels), near the top or bottom of the organ |
| p29_s75 | 0.402 | 363 | 847 | over-segmentation: the mask leaked into neighbouring tissue |
| p29_s74 | 0.495 | 745 | 867 | boundary mismatch: right place, but the outline differs from the ground truth |
| p29_s58 | 0.505 | 261 | 294 | very small tumour on this slice (261 pixels), near the top or bottom of the organ |
| p29_s69 | 0.647 | 1206 | 577 | under-segmentation: only part of the tumour was found |
| p29_s73 | 0.675 | 752 | 857 | boundary mismatch: right place, but the outline differs from the ground truth |
| p29_s59 | 0.677 | 429 | 274 | boundary mismatch: right place, but the outline differs from the ground truth |

Overlay images of the 3 worst slices are saved next to this file (red = predicted, green = ground truth).