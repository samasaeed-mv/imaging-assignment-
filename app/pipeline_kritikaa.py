"""kritikaa (TUMOUR, patient 22): liver window 80/200 -> Gaussian + CLAHE -> seeded region growing inside the
liver (tumour) -> random forest checks each tumour blob."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
import common as C
import pipeline_charvi

NAME, ORGAN, PATIENT = "kritikaa", "tumour", 22
ALGORITHMS = ("Liver window 80/200 -> Gaussian + CLAHE | Tumour: Multi-Otsu vs seeded region growing (used) | "
              "Each tumour blob checked by a random forest")
PHASE2_METRICS = ("phase2/outputs_kritikaa/phase2_metrics.csv", "regiongrow")
PHASE3_METRICS = "phase3/outputs_kritikaa/phase3_metrics.csv"
MODEL = "phase3/outputs_kritikaa/rf_model.joblib"
SE_ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def enhance(win):
    return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.GaussianBlur(win, (5, 5), 1))

def segment_tumour(enh, liver, k_dark=2.5, tolerance=30):
    roi = cv2.erode(liver.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = np.zeros(enh.shape, bool)
    if roi.sum() < 50:
        return out
    smooth = cv2.GaussianBlur(enh, (5, 5), 0)
    med, sd = np.median(smooth[roi]), smooth[roi].std()
    dark = cv2.morphologyEx(((smooth < med - k_dark * sd) & roi).astype(np.uint8), cv2.MORPH_OPEN, SE_ELLIPSE_5) > 0
    for r in regionprops(label(dark)):
        if r.area < 30:
            continue
        seed = tuple(r.coords[np.argmin(smooth[tuple(r.coords.T)])])
        grown = C.region_grow(smooth, seed, tolerance, roi)
        if grown.sum() < 0.5 * roi.sum():
            out |= grown
    return ndi.binary_fill_holes(cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, SE_ELLIPSE_5) > 0)

def run(ct, root, gt_liver=None, liver_source="ground truth"):
    bundle = joblib.load(root / MODEL)
    win = C.window(ct, 80, 200)
    enh = enhance(win)
    liver = gt_liver if (liver_source == "ground truth" and gt_liver is not None) else pipeline_charvi.predicted_liver(ct)
    tumour, checked = C.classify_regions(ct, enh, segment_tumour(enh, liver), bundle)
    return {"win": win, "enh": enh, "liver": liver, "tumour": tumour,
            "note": f"{checked} segmented blobs checked by the random forest, {label(tumour).max()} confirmed as tumour"}
