"""charvi (LIVER, patient 1): window 40/400 -> anisotropic diffusion + CLAHE -> seeded region growing (liver)
-> dark spots in the liver -> k-NN (tumour regions)."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.measure import label
import common as C

NAME, ORGAN, PATIENT = "charvi", "liver", 1
ALGORITHMS = ("Window 40/400 -> anisotropic diffusion + CLAHE | Liver: adaptive threshold vs seeded region growing (used) | "
              "Tumour regions: dark spots in liver -> k-NN")
PHASE2_METRICS = ("phase2/outputs_charvi/phase2_metrics.csv", "regiongrow")
PHASE3_METRICS = "phase3/outputs_charvi/phase3_metrics.csv"
MODEL = "phase3/outputs_charvi/knn_model.joblib"

SE_OCT_15 = np.ones((15, 15), np.uint8)
for r in range(15):
    for c in range(15):
        if min(r, 14 - r) + min(c, 14 - c) < 4:
            SE_OCT_15[r, c] = 0
SE_OCT_7 = cv2.resize(SE_OCT_15, (7, 7), interpolation=cv2.INTER_NEAREST)

def enhance(win):
    return cv2.createCLAHE(2.0, (8, 8)).apply(C.anisotropic_diffusion(win))

def segment_liver(enh):
    band = ((enh >= 150) & (enh <= 225)).astype(np.uint8)
    core = C.keep_liver(cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))))
    if not core.any():
        return np.zeros(enh.shape, bool)
    dist = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    seed = np.unravel_index(dist.argmax(), dist.shape)
    grown = C.region_grow(cv2.GaussianBlur(enh, (5, 5), 0), seed, 30)
    mask = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, SE_OCT_7) > 0
    mask = cv2.morphologyEx(C.keep_liver(mask).astype(np.uint8), cv2.MORPH_CLOSE, SE_OCT_15)
    return ndi.binary_fill_holes(mask > 0)

def predicted_liver(ct):
    """Used by the tumour members when they search inside a predicted (not ground-truth) liver."""
    return segment_liver(enhance(C.window(ct, 40, 400)))

def run(ct, root, gt_liver=None, liver_source="own"):
    bundle = joblib.load(root / MODEL)
    win = C.window(ct, 40, 400)
    enh = enhance(win)
    liver = segment_liver(enh)
    tumour, checked = C.classify_regions(ct, enh, C.dark_spots(enh, liver, k=1.5), bundle)
    return {"win": win, "enh": enh, "liver": liver, "tumour": tumour,
            "note": f"{checked} suspicious regions checked by the k-NN, {label(tumour).max()} flagged as tumour"}
