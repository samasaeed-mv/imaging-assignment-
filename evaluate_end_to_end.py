"""End-to-end test: raw CT slice -> each member's full pipeline -> final tumour mask (ground truth only for scoring).
Run from the project folder:   python evaluate_end_to_end.py
Writes report/end_to_end_per_slice.csv and report/end_to_end_results.csv."""
import glob
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))
import common  # noqa: E402

MEMBERS = ["charvi", "sama", "kritikaa", "evangeline"]


def load_slice(ct_path):
    # raw CT (HU) + ground-truth liver + ground-truth tumour
    base = ct_path[:-len("_ct.npy")]
    ct = np.load(ct_path)
    gt_liver = cv2.imread(base + "_liver.png", 0) > 0
    gt_tumour = cv2.imread(base + "_tumour.png", 0) > 0
    return ct, gt_liver, gt_tumour


def evaluate(name):
    M = importlib.import_module(f"pipeline_{name}")
    source = getattr(M, "DEFAULT_LIVER_SOURCE", "own")          # liver members find their own liver
    rows = []
    kinds = ["tumour slice", "tumour-free slice"]
    folders = [name, f"{name}_healthy"]
    for j in range(2):
        kind = kinds[j]
        folder = folders[j]
        for ct_path in sorted(glob.glob(str(ROOT / "data" / "slices" / folder / "*_ct.npy"))):
            ct, gt_liver, gt_tumour = load_slice(ct_path)
            R = M.run(ct, ROOT, gt_liver=None, liver_source=source)   # no ground truth handed to the pipeline
            pred = R["tumour"]
            row = {}
            row["member"] = name
            row["slice"] = Path(ct_path).name[:-len("_ct.npy")]
            row["kind"] = kind
            row["search area / liver"] = source
            row["liver (or search area) Dice"] = common.dice(R["liver"], gt_liver)
            if gt_tumour.any():
                row["GT tumour inside liver / search area"] = (R["liver"] & gt_tumour).sum() / max(gt_tumour.sum(), 1)
            else:
                row["GT tumour inside liver / search area"] = np.nan
            row["flagged tumour"] = bool(pred.any())
            # connected components = number of predicted tumour blobs
            row["predicted tumour blobs"] = int(cv2.connectedComponents(pred.astype(np.uint8))[0] - 1)
            row["detected (overlaps GT tumour)"] = bool((pred & gt_tumour).any())
            if gt_tumour.any():
                row["tumour Dice"] = common.dice(pred, gt_tumour)
            else:
                row["tumour Dice"] = np.nan
            rows.append(row)
            flagged_text = str(row["flagged tumour"])
            detected_text = str(row["detected (overlaps GT tumour)"])
            print(f"  {name:10s} {row['slice']:10s} {kind:17s} flagged={flagged_text:5s} "
                  f"detected={detected_text:5s}")
    return pd.DataFrame(rows)


def summarise(per_slice):
    out = []
    for name, d in per_slice.groupby("member", sort=False):
        tum = d[d.kind == "tumour slice"]
        free = d[d.kind == "tumour-free slice"]
        row = {}
        row["member"] = name
        row["search area / liver"] = d["search area / liver"].iloc[0]
        row["tumour slices"] = len(tum)
        row["tumour-free slices"] = len(free)
        row["slice detection rate (sensitivity)"] = tum["detected (overlaps GT tumour)"].mean()
        row["false-alarm slices on tumour-free slices"] = f"{int(free['flagged tumour'].sum())} / {len(free)}"
        if len(free):
            row["slice specificity (tumour-free)"] = 1 - free["flagged tumour"].mean()
            row["false-positive blobs per tumour-free slice"] = free["predicted tumour blobs"].mean()
        else:
            row["slice specificity (tumour-free)"] = np.nan
            row["false-positive blobs per tumour-free slice"] = np.nan
        row["end-to-end tumour Dice (mean)"] = tum["tumour Dice"].mean()
        row["liver / search-area Dice (mean)"] = d["liver (or search area) Dice"].mean()
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    tables = []
    for n in MEMBERS:
        tables.append(evaluate(n))
    per_slice = pd.concat(tables, ignore_index=True)
    per_slice.to_csv(ROOT / "report" / "end_to_end_per_slice.csv", index=False)
    summary = summarise(per_slice)
    summary.to_csv(ROOT / "report" / "end_to_end_results.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nEnd-to-end results (no ground truth used inside the pipeline):")
    print(summary.round(3).to_string(index=False))
    print("\nNOTE: on the tumour slices the classifier step is in-sample (the saved models were refitted on all their "
          "regions); the tumour-free slices were never used for training or tuning.")
