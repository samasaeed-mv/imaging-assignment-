"""sama (LIVER, patient 0): window 40/400 -> bilateral + CLAHE -> marker-controlled watershed (liver)
-> dark spots in the liver -> SVM (tumour regions), shape + GLCM + first-order + Hu moments."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.measure import label
import common

NAME = "sama"
ORGAN = "liver"
PATIENT = 0
ALGORITHMS = ("Window 40/400 -> bilateral + CLAHE | Liver: Multi-Otsu vs marker-controlled watershed (used) | "
              "Tumour regions: dark spots in liver -> SVM")
PHASE2_METRICS = ("phase2/outputs_sama/phase2_metrics.csv", "watershed")
PHASE3_METRICS = "phase3/outputs_sama/phase3_metrics.csv"
MODEL = "phase3/outputs_sama/svm_model.joblib"


def diamond(n):
    # Chapter 8: hand-made diamond structuring element, |row - centre| + |col - centre| <= radius
    c = n // 2
    se = np.zeros((n, n), dtype=np.uint8)
    for r in range(n):
        for q in range(n):
            if abs(r - c) + abs(q - c) <= c:
                se[r, q] = 1
    return se


SE_DIAMOND_7 = diamond(7)                                                # final opening
SE_DIAMOND_15 = diamond(15)                                              # marker separation
SE_ELLIPSE_25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # only for the solid liver core
LIVER_LOW = 150
LIVER_HIGH = 225
DARK = 110


def enhance(windowed_img):
    # bilateral filter (extension of Chapter 3), then CLAHE (adaptive extension of Chapter 2.2 HE)
    bilateral_img = cv2.bilateralFilter(windowed_img, 9, 40, 9)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(bilateral_img)


def segment_liver(enhanced_img):
    # marker-controlled watershed (extension of Chapter 7, not in the slides)
    # 1. "sure liver" marker: band threshold 150-225 (Chapter 7) + big opening (Chapter 8)
    band = ((enhanced_img >= LIVER_LOW) & (enhanced_img <= LIVER_HIGH)).astype(np.uint8)
    core = common.keep_liver(cv2.morphologyEx(band, cv2.MORPH_OPEN, SE_ELLIPSE_25))
    # 2. "sure background" marker: dark pixels + other bright organs away from the liver core
    near_core = cv2.dilate(core.astype(np.uint8), SE_DIAMOND_15).astype(bool)
    other_bright = (band.astype(bool) & ~near_core).astype(np.uint8)
    other = cv2.morphologyEx(other_bright, cv2.MORPH_OPEN, SE_DIAMOND_15) > 0
    square_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx((enhanced_img < DARK).astype(np.uint8), cv2.MORPH_OPEN, square_3) > 0
    markers = np.zeros(enhanced_img.shape, np.int32)
    markers[dark | other] = 1
    markers[core] = 2
    # 3. Gaussian smoothing + Sobel edges (Chapter 3), flood from the markers
    blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
    labels = watershed(sobel(blurred / 255.0), markers)
    # 4. clean up (Chapter 8)
    mask = cv2.morphologyEx((labels == 2).astype(np.uint8), cv2.MORPH_OPEN, SE_DIAMOND_7) > 0
    return ndi.binary_fill_holes(common.keep_liver(mask))


def hu_moment_features(mask):
    # sama's own features: the 7 Hu moment invariants of the region's shape, log-scaled
    # (extension of the Chapter 10-11 moments, not in the slides)
    hu = cv2.HuMoments(cv2.moments(mask.astype(np.uint8), binaryImage=True)).ravel()
    feats = {}
    for i in range(len(hu)):
        h = hu[i]
        if h != 0:
            feats[f"hu_{i + 1}"] = -np.sign(h) * np.log10(abs(h))
        else:
            feats[f"hu_{i + 1}"] = 0.0
    return feats


def region_features(ct, enhanced_img, mask):
    # shared features first, then the Hu moments (the saved model expects this order)
    feats = common.region_features(ct, enhanced_img, mask)
    extra = hu_moment_features(mask)
    for key in extra:
        feats[key] = extra[key]
    return feats


def run(ct, root, gt_liver=None, liver_source="own"):
    bundle = joblib.load(root / MODEL)
    windowed_img = common.window(ct, 40, 400)
    enhanced_img = enhance(windowed_img)
    liver_mask = segment_liver(enhanced_img)
    # tumour candidates: dark spots darker than the liver median - 25, then the SVM (extension of Chapter 12-13)
    candidates = common.dark_spots(enhanced_img, liver_mask, offset=25)
    tumour_mask, checked = common.classify_regions(ct, enhanced_img, candidates, bundle, features=region_features)
    result = {}
    result["win"] = windowed_img
    result["enh"] = enhanced_img
    result["liver"] = liver_mask
    result["tumour"] = tumour_mask
    result["note"] = f"{checked} suspicious regions checked by the SVM, {label(tumour_mask).max()} flagged as tumour"
    return result
