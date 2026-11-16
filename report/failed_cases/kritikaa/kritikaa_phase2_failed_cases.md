# Phase 2 failed cases: kritikaa (tumour segmentation)

Best method: **gac** (mean Dice 0.707). Slices with Dice < 0.7: 3 of 12.

| Slice | Dice (gac) | Ground-truth pixels | Predicted pixels | Likely reason |
|---|---|---|---|---|
| p22_s126 | 0.120 | 57 | 892 | very small tumour on this slice (57 pixels), near the top or bottom of the organ |
| p22_s137 | 0.221 | 153 | 472 | very small tumour on this slice (153 pixels), near the top or bottom of the organ |
| p22_s127 | 0.591 | 441 | 587 | boundary mismatch: right place, but the outline differs from the ground truth |

Overlay images of the 3 worst slices are saved next to this file (red = predicted, green = ground truth).