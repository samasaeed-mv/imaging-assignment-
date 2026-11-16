# Phase 2 failed cases: sama (liver segmentation)

Final method: **watershed** (mean Dice 0.914). Slices with Dice < 0.7: 1 of 16.

| Slice | Dice (watershed) | Ground-truth pixels | Predicted pixels | Measured reasons |
|---|---|---|---|---|
| p0_s72 | 0.679 | 4641 | 7110 | small liver cross-section (4641 px): near the dome or tip, partial volume with neighbours; step that failed: watershed leaked 3120 px into tissue of 103 HU (liver 112 HU, neighbours 45 HU): similar HU, no edge to stop at |

Diagnosis figures (overlay + HU histograms of truth, surroundings, missed and leaked pixels) for the 3 worst slices are saved next to this file.