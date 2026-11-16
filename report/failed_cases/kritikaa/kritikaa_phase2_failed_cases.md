# Phase 2 failed cases: kritikaa (tumour segmentation)

Best method: **regiongrow** (mean Dice 0.678). Slices with Dice < 0.7: 3 of 12.

| Slice | Dice (regiongrow) | Ground-truth pixels | Predicted pixels | Likely reason |
|---|---|---|---|---|
| p22_s137 | 0.000 | 153 | 254 | very small tumour on this slice (153 pixels), near the top or bottom of the organ |
| p22_s126 | 0.079 | 57 | 1393 | very small tumour on this slice (57 pixels), near the top or bottom of the organ |
| p22_s127 | 0.415 | 441 | 1571 | over-segmentation: the mask leaked into neighbouring tissue |

Overlay images of the 3 worst slices are saved next to this file (red = predicted, green = ground truth).