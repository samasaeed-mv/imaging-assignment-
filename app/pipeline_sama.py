"""sama (LIVER, patient 0): window 40/400 -> bilateral + CLAHE -> marker-controlled watershed (liver)
-> dark spots in the liver -> SVM (tumour regions), shape + GLCM + first-order + Hu moments."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.measure import label
import common as C

NAME, ORGAN, PATIENT = "sama", "liver", 0
ALGORITHMS = ("Window 40/400 -> bilateral + CLAHE | Liver: Multi-Otsu vs marker-controlled watershed (used) | "
              "Tumour regions: dark spots in liver -> SVM")
PHASE2_METRICS = ("phase2/outputs_sama/phase2_metrics.csv", "watershed")
PHASE3_METRICS = "phase3/outputs_sama/phase3_metrics.csv"
MODEL = "phase3/outputs_sama/svm_model.joblib"


def diamond(n):
    """Hand-made diamond: |row - centre| + |col - centre| <= radius."""
    c = n // 2
    return np.array([[1 if abs(r - c) + abs(q - c) <= c else 0 for q in range(n)] for r in range(n)], dtype=np.uint8)


SE_DIAMOND_7 = diamond(7)                                                # final opening
SE_DIAMOND_15 = diamond(15)                                              # marker separation
SE_ELLIPSE_25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # only for the solid liver core
LIVER_LOW, LIVER_HIGH, DARK = 150, 225, 110


def enhance(win):
    return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.bilateralFilter(win, 9, 40, 9))


def segment_liver(enh):
    band = ((enh >= LIVER_LOW) & (enh <= LIVER_HIGH)).astype(np.uint8)
    core = C.keep_liver(cv2.morphologyEx(band, cv2.MORPH_OPEN, SE_ELLIPSE_25))
    near_core = cv2.dilate(core.astype(np.uint8), SE_DIAMOND_15).astype(bool)
    other = cv2.morphologyEx((band.astype(bool) & ~near_core).astype(np.uint8), cv2.MORPH_OPEN, SE_DIAMOND_15) > 0
    dark = cv2.morphologyEx((enh < DARK).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    markers = np.zeros(enh.shape, np.int32)
    markers[dark | other] = 1
    markers[core] = 2
    labels = watershed(sobel(cv2.GaussianBlur(enh, (5, 5), 0) / 255.0), markers)
    mask = cv2.morphologyEx((labels == 2).astype(np.uint8), cv2.MORPH_OPEN, SE_DIAMOND_7) > 0
    return ndi.binary_fill_holes(C.keep_liver(mask))


def hu_moment_features(mask):
    """sama's own features: the 7 Hu moment invariants of the region's shape, log-scaled."""
    hu = cv2.HuMoments(cv2.moments(mask.astype(np.uint8), binaryImage=True)).ravel()
    return {f"hu_{i + 1}": -np.sign(h) * np.log10(abs(h)) if h != 0 else 0.0 for i, h in enumerate(hu)}


def region_features(ct, enh, mask):
    return {**C.region_features(ct, enh, mask), **hu_moment_features(mask)}


def run(ct, root, gt_liver=None, liver_source="own"):
    bundle = joblib.load(root / MODEL)
    win = C.window(ct, 40, 400)
    enh = enhance(win)
    liver = segment_liver(enh)
    tumour, checked = C.classify_regions(ct, enh, C.dark_spots(enh, liver, offset=25), bundle, features=region_features)
    return {"win": win, "enh": enh, "liver": liver, "tumour": tumour,
            "note": f"{checked} suspicious regions checked by the SVM, {label(tumour).max()} flagged as tumour"}
