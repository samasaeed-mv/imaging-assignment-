"""kritikaa (TUMOUR, patient 22): liver window 80/200 -> non-local means + CLAHE -> geodesic active contour inside
the liver (tumour) -> random forest checks each tumour blob (shape + GLCM + first-order + border sharpness)."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
import common
import pipeline_charvi

NAME = "kritikaa"
ORGAN = "tumour"
PATIENT = 22
ALGORITHMS = ("Liver window 80/200 -> non-local means + CLAHE | Tumour: Niblack vs statistical region growing vs "
              "geodesic active contour (used) | Each tumour blob checked by a random forest")
PHASE2_METRICS = ("phase2/outputs_kritikaa/phase2_metrics.csv", "gac")
PHASE3_METRICS = "phase3/outputs_kritikaa/phase3_metrics.csv"
MODEL = "phase3/outputs_kritikaa/rf_model.joblib"
# end-to-end search area used by default in the app: the better of the two automatic ones in Phase 2
# (tumour Dice: predicted liver 0.537, convex hull of predicted liver 0.546)
DEFAULT_LIVER_SOURCE = "convex hull of predicted liver"

# hand-made plus structuring elements (same as the Phase 2 notebook)
SE_PLUS_5 = np.zeros((5, 5), np.uint8)
SE_PLUS_5[2, :] = 1
SE_PLUS_5[:, 2] = 1
SE_PLUS_5[1:4, 1:4] = 1
SE_PLUS_7 = np.zeros((7, 7), np.uint8)
SE_PLUS_7[3, :] = 1
SE_PLUS_7[:, 3] = 1
SE_PLUS_7[2:5, 2:5] = 1
K_DARK = 2.0
GAC_ITERATIONS = 50
GAC_ALPHA = 100
GAC_SIGMA = 2


def enhance(windowed_img):
    # non-local means, then CLAHE
    nlm_img = cv2.fastNlMeansDenoising(windowed_img, None, h=20, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(nlm_img)


def liver_roi(liver_mask):
    # erode the liver a little (5x5 square) so its dark border is not mistaken for tumour
    square_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.erode(liver_mask.astype(np.uint8), square_5) > 0


def remove_small(mask, min_area=30):
    # connected components -> keep the blobs with at least min_area pixels
    lab = label(mask)
    keep = []
    for r in regionprops(lab):
        if r.area >= min_area:
            keep.append(r.label)
    return np.isin(lab, keep)


def tidy(mask):
    # opening (plus 5), drop tiny blobs, closing (plus 7), fill holes
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, SE_PLUS_5) > 0
    mask = remove_small(mask)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, SE_PLUS_7) > 0
    return ndi.binary_fill_holes(mask)


def dark_candidates(enhanced_img, liver_mask):
    # dark spots in the liver (darker than median - 2 x std): the starting curve of the active contour
    roi = liver_roi(liver_mask)
    if roi.sum() < 50:
        return np.zeros(enhanced_img.shape, bool)
    smooth = cv2.GaussianBlur(enhanced_img, (5, 5), 0)   # Gaussian smoothing
    med = np.median(smooth[roi])
    sd = smooth[roi].std()
    dark = ((smooth < med - K_DARK * sd) & roi).astype(np.uint8)   # threshold
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, SE_PLUS_5) > 0
    return remove_small(dark)


def segment_tumour(enhanced_img, liver_mask):
    # geodesic active contour
    # grow the dark candidates outwards until the curve meets a strong edge
    start = dark_candidates(enhanced_img, liver_mask)
    if not start.any():
        return np.zeros(enhanced_img.shape, bool)
    edge_stop = inverse_gaussian_gradient(enhanced_img / 255.0, alpha=GAC_ALPHA, sigma=GAC_SIGMA)
    level_set = morphological_geodesic_active_contour(edge_stop, num_iter=GAC_ITERATIONS, init_level_set=start.astype(np.int8),
                                                     smoothing=1, balloon=1, threshold="auto")
    return tidy((level_set > 0) & liver_roi(liver_mask))


def border_features(ct, enhanced_img, mask):
    # kritikaa's own features: border sharpness (HU step, edge strength)
    # uses dilation / erosion and the Sobel gradient
    m8 = mask.astype(np.uint8)
    ring = cv2.dilate(m8, SE_PLUS_7, iterations=2).astype(bool) & ~mask
    edge = mask & ~(cv2.erode(m8, SE_PLUS_5) > 0)
    gx = cv2.Sobel(enhanced_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced_img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    if edge.any():
        g = grad[edge]
    else:
        g = grad[mask]
    feats = {}
    feats["border_HU_step"] = ct[ring].mean() - ct[mask].mean()
    feats["border_gradient_mean"] = g.mean()
    feats["border_gradient_cv"] = g.std() / (g.mean() + 1e-9)
    return feats


def region_features(ct, enhanced_img, mask):
    # shared features first, then border sharpness (the saved model expects this order)
    feats = common.region_features(ct, enhanced_img, mask)
    extra = border_features(ct, enhanced_img, mask)
    for key in extra:
        feats[key] = extra[key]
    return feats


def run(ct, root, gt_liver=None, liver_source=DEFAULT_LIVER_SOURCE):
    bundle = joblib.load(root / MODEL)
    windowed_img = common.window(ct, 80, 200)
    enhanced_img = enhance(windowed_img)
    liver_mask = pipeline_charvi.search_area(ct, gt_liver, liver_source)
    # every segmented blob is checked by the random forest
    blobs = segment_tumour(enhanced_img, liver_mask)
    tumour_mask, checked = common.classify_regions(ct, enhanced_img, blobs, bundle, features=region_features)
    result = {}
    result["win"] = windowed_img
    result["enh"] = enhanced_img
    result["liver"] = liver_mask
    result["tumour"] = tumour_mask
    result["note"] = f"{checked} segmented blobs checked by the random forest, {label(tumour_mask).max()} confirmed as tumour"
    return result
