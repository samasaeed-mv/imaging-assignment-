"""evangeline (TUMOUR, patient 29): narrow window 90/150 -> guided filter (r 8, eps 0.05) -> Chan-Vese inside the
liver (tumour) -> decision tree on SLIC superpixels (diagnosis), shape + GLCM + first-order + Laws texture energy."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.segmentation import morphological_chan_vese, slic
from skimage.measure import regionprops, label
import common as C
import pipeline_charvi

NAME, ORGAN, PATIENT = "evangeline", "tumour", 29
ALGORITHMS = ("Window 90/150 -> guided filter | Tumour: Sauvola vs marker watershed (distance-transform markers) vs "
              "Chan-Vese (used) | Diagnosis: decision tree on SLIC superpixels")
PHASE2_METRICS = ("phase2/outputs_evangeline/phase2_metrics.csv", "chanvese")
PHASE3_METRICS = "phase3/outputs_evangeline/phase3_metrics.csv"
MODEL = "phase3/outputs_evangeline/tree_model.joblib"


def hexagon(n):
    """Hand-made hexagon structuring element (flat top/bottom, slanted sides)."""
    se, c = np.zeros((n, n), np.uint8), n // 2
    q = (n + 1) // 4
    for r in range(n):
        cut = max(0, abs(r - c) - (c - q))
        se[r, cut:n - cut] = 1
    return se


SE_HEX_5, SE_HEX_9 = hexagon(5), hexagon(9)
CV_ITERATIONS, CV_START_K = 50, 0.75
L5, E5, S5, R5 = [1, 4, 6, 4, 1], [-1, -2, 0, 2, 1], [-1, 0, 2, 0, -1], [1, -4, 6, -4, 1]
LAWS_PAIRS = {"L5E5": (L5, E5), "L5S5": (L5, S5), "E5E5": (E5, E5), "S5S5": (S5, S5), "R5R5": (R5, R5)}


def guided_filter(img, r=8, eps=0.05):
    """Guided filter (He et al.) with box filters; the image guides itself."""
    I = img.astype(np.float64) / 255
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mean_I = box(I)
    var_I = box(I * I) - mean_I ** 2
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I
    return np.clip((box(a) * I + box(b)) * 255, 0, 255).astype(np.uint8)


def enhance(win):
    return guided_filter(win)


def liver_roi(liver):
    return cv2.erode(liver.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def clean(mask, min_area=30):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, SE_HEX_5) > 0
    lab = label(mask)
    mask = np.isin(lab, [r.label for r in regionprops(lab) if r.area >= min_area])
    return ndi.binary_fill_holes(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, SE_HEX_9) > 0)


def segment_tumour(enh, liver):
    """Chan-Vese inside the liver's bounding box, started from the dark candidate area; keep the darker region."""
    roi = liver_roi(liver)
    if roi.sum() < 50:
        return np.zeros(enh.shape, bool)
    smooth = cv2.GaussianBlur(enh, (5, 5), 0).astype(float) / 255
    ys, xs = np.nonzero(roi)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, inside = smooth[y0:y1, x0:x1].copy(), roi[y0:y1, x0:x1]
    med = np.median(crop[inside])
    crop[~inside] = med
    start = (crop < med - CV_START_K * crop[inside].std()) & inside
    if not start.any():
        return np.zeros(enh.shape, bool)
    level_set = morphological_chan_vese(crop, num_iter=CV_ITERATIONS, init_level_set=start.astype(np.int8),
                                        smoothing=1, lambda1=1, lambda2=1).astype(bool)
    if (level_set & inside).any() and (~level_set & inside).any() and crop[level_set & inside].mean() > crop[~level_set & inside].mean():
        level_set = ~level_set
    out = np.zeros(enh.shape, bool); out[y0:y1, x0:x1] = level_set & inside
    return clean(out)


def laws_features(enh, mask):
    """evangeline's own features: Laws texture energy (local brightness removed, mean |response| in the region)."""
    img = enh.astype(np.float64) - cv2.blur(enh.astype(np.float64), (15, 15))
    feats = {}
    for name, (a, b) in LAWS_PAIRS.items():
        k = np.outer(a, b).astype(np.float64)
        response = np.abs(cv2.filter2D(img, -1, k))
        if a != b:
            response = (response + np.abs(cv2.filter2D(img, -1, k.T))) / 2
        feats[f"laws_{name}"] = response[mask].mean()
    return feats


def region_features(ct, enh, mask):
    return {**C.region_features(ct, enh, mask), **laws_features(enh, mask)}


def diagnose(ct, enh, liver, bundle):
    """Cut the liver into ~20 superpixels and let the decision tree label each one. Returns the flagged area."""
    model, names = bundle["model"], bundle["feature_names"]
    flagged_area = np.zeros(enh.shape, bool)
    if liver.sum() < 50:
        return 0, 0, flagged_area
    sp = slic(enh, n_segments=20, compactness=0.1, mask=liver, channel_axis=None, start_label=1)
    checked = flagged = 0
    for i in range(1, sp.max() + 1):
        m = sp == i
        if m.sum() < 10:
            continue
        f = region_features(ct, enh, m)
        checked += 1
        if model.predict(np.array([[f[n] for n in names]]))[0] == 1:
            flagged += 1; flagged_area |= m
    return checked, flagged, flagged_area


def run(ct, root, gt_liver=None, liver_source="ground truth"):
    bundle = joblib.load(root / MODEL)
    win = C.window(ct, 90, 150)
    enh = enhance(win)
    liver = gt_liver if (liver_source == "ground truth" and gt_liver is not None) else pipeline_charvi.predicted_liver(ct)
    segmented = segment_tumour(enh, liver)
    checked, flagged, flagged_area = diagnose(ct, enh, liver, bundle)
    # keep only the segmented tumour blobs that touch a superpixel the decision tree called tumour
    lab = label(segmented)
    keep = np.unique(lab[flagged_area & (lab > 0)])
    tumour = np.isin(lab, keep[keep > 0])
    return {"win": win, "enh": enh, "liver": liver, "tumour": tumour,
            "note": f"{checked} liver superpixels checked by the decision tree, {flagged} flagged as tumour"}
