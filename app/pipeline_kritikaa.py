"""kritikaa (TUMOUR, patient 22): liver window 80/200 -> non-local means + CLAHE -> geodesic active contour inside
the liver (tumour) -> random forest checks each tumour blob (shape + GLCM + first-order + border sharpness)."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
import common as C
import pipeline_charvi

NAME, ORGAN, PATIENT = "kritikaa", "tumour", 22
ALGORITHMS = ("Liver window 80/200 -> non-local means + CLAHE | Tumour: Niblack vs statistical region growing vs "
              "geodesic active contour (used) | Each tumour blob checked by a random forest")
PHASE2_METRICS = ("phase2/outputs_kritikaa/phase2_metrics.csv", "gac")
PHASE3_METRICS = "phase3/outputs_kritikaa/phase3_metrics.csv"
MODEL = "phase3/outputs_kritikaa/rf_model.joblib"

# hand-made plus structuring elements (same as the Phase 2 notebook)
SE_PLUS_5 = np.zeros((5, 5), np.uint8); SE_PLUS_5[2, :] = 1; SE_PLUS_5[:, 2] = 1; SE_PLUS_5[1:4, 1:4] = 1
SE_PLUS_7 = np.zeros((7, 7), np.uint8); SE_PLUS_7[3, :] = 1; SE_PLUS_7[:, 3] = 1; SE_PLUS_7[2:5, 2:5] = 1
K_DARK, GAC_ITERATIONS, GAC_ALPHA, GAC_SIGMA = 2.0, 50, 100, 2


def enhance(win):
    """Non-local means (h = 20) then CLAHE."""
    nlm = cv2.fastNlMeansDenoising(win, None, h=20, templateWindowSize=7, searchWindowSize=21)
    return cv2.createCLAHE(2.0, (8, 8)).apply(nlm)


def liver_roi(liver):
    return cv2.erode(liver.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def remove_small(mask, min_area=30):
    lab = label(mask)
    return np.isin(lab, [r.label for r in regionprops(lab) if r.area >= min_area])


def tidy(mask):
    mask = remove_small(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, SE_PLUS_5) > 0)
    return ndi.binary_fill_holes(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, SE_PLUS_7) > 0)


def dark_candidates(enh, liver):
    """Dark spots in the liver (darker than median - 2 x std): the starting curve of the active contour."""
    roi = liver_roi(liver)
    if roi.sum() < 50:
        return np.zeros(enh.shape, bool)
    smooth = cv2.GaussianBlur(enh, (5, 5), 0)
    med, sd = np.median(smooth[roi]), smooth[roi].std()
    dark = cv2.morphologyEx(((smooth < med - K_DARK * sd) & roi).astype(np.uint8), cv2.MORPH_OPEN, SE_PLUS_5) > 0
    return remove_small(dark)


def segment_tumour(enh, liver):
    """Geodesic active contour: grow the dark candidates outwards until the curve meets a strong edge."""
    start = dark_candidates(enh, liver)
    if not start.any():
        return np.zeros(enh.shape, bool)
    edge_stop = inverse_gaussian_gradient(enh / 255.0, alpha=GAC_ALPHA, sigma=GAC_SIGMA)
    level_set = morphological_geodesic_active_contour(edge_stop, num_iter=GAC_ITERATIONS, init_level_set=start.astype(np.int8),
                                                     smoothing=1, balloon=1, threshold="auto")
    return tidy((level_set > 0) & liver_roi(liver))


def border_features(ct, enh, mask):
    """kritikaa's own features: how sharp the region's border is (HU step to the surroundings, edge strength)."""
    m8 = mask.astype(np.uint8)
    ring = cv2.dilate(m8, SE_PLUS_7, iterations=2).astype(bool) & ~mask
    edge = mask & ~(cv2.erode(m8, SE_PLUS_5) > 0)
    grad = np.hypot(cv2.Sobel(enh, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(enh, cv2.CV_64F, 0, 1, ksize=3))
    g = grad[edge] if edge.any() else grad[mask]
    return {"border_HU_step": ct[ring].mean() - ct[mask].mean(), "border_gradient_mean": g.mean(),
            "border_gradient_cv": g.std() / (g.mean() + 1e-9)}


def region_features(ct, enh, mask):
    return {**C.region_features(ct, enh, mask), **border_features(ct, enh, mask)}


def run(ct, root, gt_liver=None, liver_source="ground truth"):
    bundle = joblib.load(root / MODEL)
    win = C.window(ct, 80, 200)
    enh = enhance(win)
    liver = gt_liver if (liver_source == "ground truth" and gt_liver is not None) else pipeline_charvi.predicted_liver(ct)
    tumour, checked = C.classify_regions(ct, enh, segment_tumour(enh, liver), bundle, features=region_features)
    return {"win": win, "enh": enh, "liver": liver, "tumour": tumour,
            "note": f"{checked} segmented blobs checked by the random forest, {label(tumour).max()} confirmed as tumour"}
