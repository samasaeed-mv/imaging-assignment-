"""End-to-end evaluation: raw CT slice -> each member's full pipeline (app/pipeline_<name>.py) -> final tumour mask.

No ground truth is used inside the pipeline: the liver members find their own liver, the tumour members search in
their default automatic search area (predicted liver or its convex hull). Ground truth is only used for scoring.

Run from the project folder:   python evaluate_end_to_end.py
Writes report/end_to_end_per_slice.csv and report/end_to_end_results.csv.

Slice-level scores:
  - tumour slice detected   = the final mask overlaps the ground-truth tumour
  - false-alarm slice       = a tumour-free slice where the pipeline flags any tumour
NOTE: the saved Phase 3 classifiers were refitted on ALL regions of the tumour slices, so on the tumour slices the
classifier step is in-sample (optimistic). The tumour-free slices (data/slices/<name>_healthy) were never used for
training or tuning, so their false-alarm rate is a fair estimate.
"""
import glob
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))
import common as C  # noqa: E402

MEMBERS = ["charvi", "sama", "kritikaa", "evangeline"]


def load_slice(ct_path):
    base = ct_path[:-len("_ct.npy")]
    return np.load(ct_path), cv2.imread(base + "_liver.png", 0) > 0, cv2.imread(base + "_tumour.png", 0) > 0


def evaluate(name):
    M = importlib.import_module(f"pipeline_{name}")
    source = getattr(M, "DEFAULT_LIVER_SOURCE", "own")          # liver members find their own liver
    rows = []
    for kind, folder in [("tumour slice", name), ("tumour-free slice", f"{name}_healthy")]:
        for ct_path in sorted(glob.glob(str(ROOT / "data" / "slices" / folder / "*_ct.npy"))):
            ct, gt_liver, gt_tumour = load_slice(ct_path)
            R = M.run(ct, ROOT, gt_liver=None, liver_source=source)   # no ground truth handed to the pipeline
            pred = R["tumour"]
            rows.append({"member": name, "slice": Path(ct_path).name[:-len("_ct.npy")], "kind": kind,
                         "search area / liver": source,
                         "liver (or search area) Dice": C.dice(R["liver"], gt_liver),
                         "GT tumour inside liver / search area": (R["liver"] & gt_tumour).sum() / max(gt_tumour.sum(), 1)
                         if gt_tumour.any() else np.nan,
                         "flagged tumour": bool(pred.any()), "predicted tumour blobs": int(cv2.connectedComponents(pred.astype(np.uint8))[0] - 1),
                         "detected (overlaps GT tumour)": bool((pred & gt_tumour).any()),
                         "tumour Dice": C.dice(pred, gt_tumour) if gt_tumour.any() else np.nan})
            print(f"  {name:10s} {rows[-1]['slice']:10s} {kind:17s} flagged={rows[-1]['flagged tumour']!s:5s} "
                  f"detected={rows[-1]['detected (overlaps GT tumour)']!s:5s}")
    return pd.DataFrame(rows)


def summarise(per_slice):
    out = []
    for name, d in per_slice.groupby("member", sort=False):
        tum, free = d[d.kind == "tumour slice"], d[d.kind == "tumour-free slice"]
        out.append({"member": name, "search area / liver": d["search area / liver"].iloc[0],
                    "tumour slices": len(tum), "tumour-free slices": len(free),
                    "slice detection rate (sensitivity)": tum["detected (overlaps GT tumour)"].mean(),
                    "false-alarm slices on tumour-free slices": f"{int(free['flagged tumour'].sum())} / {len(free)}",
                    "slice specificity (tumour-free)": 1 - free["flagged tumour"].mean() if len(free) else np.nan,
                    "false-positive blobs per tumour-free slice": free["predicted tumour blobs"].mean() if len(free) else np.nan,
                    "end-to-end tumour Dice (mean)": tum["tumour Dice"].mean(),
                    "liver / search-area Dice (mean)": d["liver (or search area) Dice"].mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    per_slice = pd.concat([evaluate(n) for n in MEMBERS], ignore_index=True)
    per_slice.to_csv(ROOT / "report" / "end_to_end_per_slice.csv", index=False)
    summary = summarise(per_slice)
    summary.to_csv(ROOT / "report" / "end_to_end_results.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nEnd-to-end results (no ground truth used inside the pipeline):")
    print(summary.round(3).to_string(index=False))
    print("\nNOTE: on the tumour slices the classifier step is in-sample (the saved models were refitted on all their "
          "regions); the tumour-free slices were never used for training or tuning.")
