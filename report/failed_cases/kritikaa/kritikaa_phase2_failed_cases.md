# Phase 2 failed cases: kritikaa (tumour segmentation)

Final method: **gac** (mean Dice 0.707). Slices with Dice < 0.7: 3 of 12.

| Slice | Dice (gac) | Ground-truth pixels | Predicted pixels | Measured reasons |
|---|---|---|---|---|
| p22_s126 | 0.120 | 57 | 892 | small tumour (57 px): partial-volume effect, its edge HU blends into the liver; tumour 55 HU darker than liver, liver noise 29 HU (contrast/noise 1.92); step that failed: the active contour leaked 835 px into tissue of 55 HU (tumour 50 HU, neighbours 105 HU): similar HU, no edge to stop at |
| p22_s137 | 0.221 | 153 | 472 | small tumour (153 px): partial-volume effect, its edge HU blends into the liver; tumour 34 HU darker than liver, liver noise 26 HU (contrast/noise 1.31); step that failed: the active contour leaked 403 px into tissue of 59 HU (tumour 68 HU, neighbours 102 HU): similar HU, no edge to stop at |
| p22_s127 | 0.591 | 441 | 587 | tumour 54 HU darker than liver, liver noise 26 HU (contrast/noise 2.08); step that failed: the active contour leaked 283 px into tissue of 51 HU (tumour 51 HU, neighbours 105 HU): similar HU, no edge to stop at |

Diagnosis figures (overlay + HU histograms of truth, surroundings, missed and leaked pixels) for the 3 worst slices are saved next to this file.