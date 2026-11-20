"""evangeline (TUMOUR, patient 29): narrow window 90/150 -> guided filter (r 8, eps 0.05) -> Sauvola threshold inside
the liver (tumour) -> decision tree on SLIC superpixels (diagnosis), shape + GLCM + first-order + Laws texture energy."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.filters import threshold_sauvola
from skimage.segmentation import slic
from skimage.measure import regionprops, label
import common
import pipeline_charvi

NAME = "evangeline"
ORGAN = "tumour"
PATIENT = 29
ALGORITHMS = ("Window 90/150 -> guided filter | Tumour: Sauvola (used) vs marker watershed (distance-transform markers) vs "
              "Chan-Vese | Diagnosis: decision tree on SLIC superpixels")
PHASE2_METRICS = ("phase2/outputs_evangeline/phase2_metrics.csv", "sauvola")
PHASE3_METRICS = "phase3/outputs_evangeline/phase3_metrics.csv"
MODEL = "phase3/outputs_evangeline/tree_model.joblib"
# end-to-end search area used by default in the app: the better of the two automatic ones in Phase 2
# (tumour Dice: predicted liver 0.199, convex hull of predicted liver 0.178)
DEFAULT_LIVER_SOURCE = "predicted liver"


def hexagon(n):
    # Chapter 8: hand-made hexagon structuring element (flat top/bottom, slanted sides)
    se = np.zeros((n, n), np.uint8)
    c = n // 2
    q = (n + 1) // 4
    for r in range(n):
        cut = max(0, abs(r - c) - (c - q))
        se[r, cut:n - cut] = 1
    return se


SE_HEX_5 = hexagon(5)
SE_HEX_9 = hexagon(9)
SAUVOLA_WINDOW = 101
SAUVOLA_K = 0.1
SAUVOLA_R = 128

# Laws' 1D masks: Level, Edge, Spot, Ripple. A 2D mask = outer product of two of them
L5 = [1, 4, 6, 4, 1]
E5 = [-1, -2, 0, 2, 1]
S5 = [-1, 0, 2, 0, -1]
R5 = [1, -4, 6, -4, 1]
LAWS_PAIRS = {}
LAWS_PAIRS["L5E5"] = (L5, E5)
LAWS_PAIRS["L5S5"] = (L5, S5)
LAWS_PAIRS["E5E5"] = (E5, E5)
LAWS_PAIRS["S5S5"] = (S5, S5)
LAWS_PAIRS["R5R5"] = (R5, R5)


def guided_filter(img, r=8, eps=0.05):
    # guided filter (He et al., extension of the Chapter 3 average filters) with box filters; the image guides itself
    size = (2 * r + 1, 2 * r + 1)
    I = img.astype(np.float64) / 255
    mean_I = cv2.boxFilter(I, -1, size)
    var_I = cv2.boxFilter(I * I, -1, size) - mean_I ** 2
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I
    q = cv2.boxFilter(a, -1, size) * I + cv2.boxFilter(b, -1, size)
    return np.clip(q * 255, 0, 255).astype(np.uint8)


def enhance(windowed_img):
    # no contrast step for this patient: HE and CLAHE lowered the tumour CNR
    return guided_filter(windowed_img)


def liver_roi(liver_mask):
    # Chapter 8: erode the liver a little (5x5 square) so its dark border is not mistaken for tumour
    square_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.erode(liver_mask.astype(np.uint8), square_5) > 0


def clean(mask, min_area=30):
    # Chapter 8: opening (hexagon 5), drop small blobs, closing (hexagon 9), fill holes
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, SE_HEX_5) > 0
    lab = label(mask)
    keep = []
    for r in regionprops(lab):
        if r.area >= min_area:
            keep.append(r.label)
    mask = np.isin(lab, keep)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, SE_HEX_9) > 0
    return ndi.binary_fill_holes(mask)


def segment_tumour(enhanced_img, liver_mask):
    # Sauvola local threshold inside the liver (extension of the Chapter 7 adaptive thresholding, slide 23):
    # T = m x (1 + k x (s / R - 1)); pixels darker than T are tumour
    roi = liver_roi(liver_mask)
    if roi.sum() < 50:
        return np.zeros(enhanced_img.shape, bool)
    smooth = cv2.GaussianBlur(enhanced_img, (5, 5), 0).astype(float)
    filled = smooth.copy()
    filled[~roi] = np.median(smooth[roi])          # outside the liver = liver median
    T = threshold_sauvola(filled, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K, r=SAUVOLA_R)
    return clean((smooth < T) & roi)


def laws_features(enhanced_img, mask):
    # evangeline's own features: Laws texture energy (extension of the Chapter 10-11 texture descriptors);
    # each Laws mask is a 5x5 convolution mask (Chapter 3). Local brightness removed, mean |response| in the region
    img = enhanced_img.astype(np.float64) - cv2.blur(enhanced_img.astype(np.float64), (15, 15))
    feats = {}
    for name in LAWS_PAIRS:
        a, b = LAWS_PAIRS[name]
        k = np.outer(a, b).astype(np.float64)
        response = np.abs(cv2.filter2D(img, -1, k))
        if a != b:
            response = (response + np.abs(cv2.filter2D(img, -1, k.T))) / 2
        feats[f"laws_{name}"] = response[mask].mean()
    return feats


def region_features(ct, enhanced_img, mask):
    # shared features first, then Laws (the saved model expects this order)
    feats = common.region_features(ct, enhanced_img, mask)
    extra = laws_features(enhanced_img, mask)
    for key in extra:
        feats[key] = extra[key]
    return feats


def diagnose(ct, enhanced_img, liver_mask, bundle):
    # cut the liver into ~20 SLIC superpixels (extension of Chapter 7) and let the decision tree
    # (extension of Chapter 12-13) label each one. Returns the flagged area
    model = bundle["model"]
    names = bundle["feature_names"]
    flagged_area = np.zeros(enhanced_img.shape, bool)
    if liver_mask.sum() < 50:
        return 0, 0, flagged_area
    sp = slic(enhanced_img, n_segments=20, compactness=0.1, mask=liver_mask, channel_axis=None, start_label=1)
    checked = 0
    flagged = 0
    for i in range(1, sp.max() + 1):
        m = sp == i
        if m.sum() < 10:
            continue
        f = region_features(ct, enhanced_img, m)
        checked += 1
        row = []
        for n in names:
            row.append(f[n])
        if model.predict(np.array([row]))[0] == 1:
            flagged += 1
            flagged_area |= m
    return checked, flagged, flagged_area


def run(ct, root, gt_liver=None, liver_source=DEFAULT_LIVER_SOURCE):
    bundle = joblib.load(root / MODEL)
    windowed_img = common.window(ct, 90, 150)
    enhanced_img = enhance(windowed_img)
    liver_mask = pipeline_charvi.search_area(ct, gt_liver, liver_source)
    segmented = segment_tumour(enhanced_img, liver_mask)
    checked, flagged, flagged_area = diagnose(ct, enhanced_img, liver_mask, bundle)
    # keep only the segmented tumour blobs that touch a superpixel the decision tree called tumour
    lab = label(segmented)
    keep = np.unique(lab[flagged_area & (lab > 0)])
    tumour_mask = np.isin(lab, keep[keep > 0])
    result = {}
    result["win"] = windowed_img
    result["enh"] = enhanced_img
    result["liver"] = liver_mask
    result["tumour"] = tumour_mask
    result["note"] = f"{checked} liver superpixels checked by the decision tree, {flagged} flagged as tumour"
    return result
