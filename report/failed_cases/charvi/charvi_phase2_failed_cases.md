# Phase 2 failed cases: charvi (liver segmentation)

Final method: **regiongrow** (mean Dice 0.886). Slices with Dice < 0.7: 1 of 18.

| Slice | Dice (regiongrow) | Ground-truth pixels | Predicted pixels | Measured reasons |
|---|---|---|---|---|
| p1_s73 | 0.000 | 211 | 1831 | small liver cross-section (211 px): near the dome or tip, partial volume with neighbours; step that failed: liver core / centroid seed (nothing started on the liver) |

Diagnosis figures (overlay + HU histograms of truth, surroundings, missed and leaked pixels) for the 3 worst slices are saved next to this file.