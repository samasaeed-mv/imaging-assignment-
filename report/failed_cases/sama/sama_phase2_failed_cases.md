# Phase 2 failed cases: sama (liver segmentation)

Best method: **watershed** (mean Dice 0.913). Slices with Dice < 0.7: 1 of 16.

| Slice | Dice (watershed) | Ground-truth pixels | Predicted pixels | Likely reason |
|---|---|---|---|---|
| p0_s72 | 0.678 | 4641 | 7110 | over-segmentation: the mask leaked into neighbouring tissue |

Overlay images of the 3 worst slices are saved next to this file (red = predicted, green = ground truth).