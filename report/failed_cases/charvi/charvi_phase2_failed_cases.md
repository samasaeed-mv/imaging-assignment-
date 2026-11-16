# Phase 2 failed cases: charvi (liver segmentation)

Best method: **regiongrow** (mean Dice 0.880). Slices with Dice < 0.7: 1 of 18.

| Slice | Dice (regiongrow) | Ground-truth pixels | Predicted pixels | Likely reason |
|---|---|---|---|---|
| p1_s73 | 0.000 | 211 | 1850 | very small liver on this slice (211 pixels), near the top or bottom of the organ |

Overlay images of the 3 worst slices are saved next to this file (red = predicted, green = ground truth).