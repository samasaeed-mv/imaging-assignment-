"""charvi (LIVER, patient 1): window 40/400 -> anisotropic diffusion + CLAHE -> seeded region growing from the liver
core's centroid + dark tumour inside the liver outline (liver) -> dark spots in the liver -> k-NN (tumour regions)."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import convex_hull_image
from skimage.feature import local_binary_pattern
import common

NAME = "charvi"
ORGAN = "liver"
PATIENT = 1
ALGORITHMS = ("Window 40/400 -> anisotropic diffusion + CLAHE | Liver: adaptive threshold vs seeded region growing (used) | "
              "Tumour regions: dark spots in liver -> k-NN")
PHASE2_METRICS = ("phase2/outputs_charvi/phase2_metrics.csv", "regiongrow")
PHASE3_METRICS = "phase3/outputs_charvi/phase3_metrics.csv"
MODEL = "phase3/outputs_charvi/knn_model.joblib"

# Chapter 8: hand-made octagon structuring element (square with cut corners), in 3 sizes
SE_OCT_15 = np.ones((15, 15), np.uint8)
for r in range(15):
    for c in range(15):
        if min(r, 14 - r) + min(c, 14 - c) < 4:
            SE_OCT_15[r, c] = 0
SE_OCT_7 = cv2.resize(SE_OCT_15, (7, 7), interpolation=cv2.INTER_NEAREST)
SE_OCT_61 = cv2.resize(SE_OCT_15, (61, 61), interpolation=cv2.INTER_NEAREST)
LIVER_LOW = 150
LIVER_HIGH = 225
TOLERANCE = 30
FAT_LEVEL = 60


def anisotropic_diffusion(img, iterations=15, kappa=30, gamma=0.15):
    # Perona-Malik diffusion (extension of the Chapter 3 smoothing filters): smooths flat areas but stops at edges
    I = img.astype(np.float64)
    for _ in range(iterations):
        diffs = [np.roll(I, -1, 0) - I, np.roll(I, 1, 0) - I, np.roll(I, -1, 1) - I, np.roll(I, 1, 1) - I]
        flow = 0
        for d in diffs:
            flow = flow + np.exp(-(d / kappa) ** 2) * d
        I = I + gamma * flow
    return np.clip(I, 0, 255).astype(np.uint8)


def region_grow(img, seed, tolerance, allowed=None):
    # seeded region growing with a running-average brightness tolerance (extension of Chapter 7)
    h, w = img.shape
    region = np.zeros((h, w), bool)
    region[seed] = True
    stack = [seed]
    total = float(img[seed])
    count = 1
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = y + dy
            xx = x + dx
            if 0 <= yy < h and 0 <= xx < w and not region[yy, xx] and (allowed is None or allowed[yy, xx]):
                if abs(float(img[yy, xx]) - total / count) <= tolerance:
                    region[yy, xx] = True
                    stack.append((yy, xx))
                    total += float(img[yy, xx])
                    count += 1
    return region


def enhance(windowed_img):
    # anisotropic diffusion, then CLAHE (adaptive extension of Chapter 2.2 HE)
    aniso_img = anisotropic_diffusion(windowed_img)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(aniso_img)


def find_seed(enhanced_img):
    # seed = spatial centroid of the liver core (nearest core pixel if the centroid is outside the core)
    band = ((enhanced_img >= LIVER_LOW) & (enhanced_img <= LIVER_HIGH)).astype(np.uint8)
    ellipse_25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    core = common.keep_liver(cv2.morphologyEx(band, cv2.MORPH_OPEN, ellipse_25))
    if not core.any():
        return None
    ys, xs = np.nonzero(core)
    i = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
    return ys[i], xs[i]


def add_tumour(enhanced_img, liver_mask):
    # take in the dark tumour region growing leaves out: tumour-grey pixels inside the closed liver outline
    closed = cv2.morphologyEx(liver_mask.astype(np.uint8), cv2.MORPH_CLOSE, SE_OCT_61) > 0
    outline = ndi.binary_fill_holes(closed)
    dark = outline & ~liver_mask & (enhanced_img >= FAT_LEVEL) & (enhanced_img < LIVER_LOW)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, SE_OCT_7) > 0
    joined = cv2.morphologyEx((liver_mask | dark).astype(np.uint8), cv2.MORPH_CLOSE, SE_OCT_15) > 0
    return common.keep_liver(ndi.binary_fill_holes(joined))


def segment_liver(enhanced_img):
    seed = find_seed(enhanced_img)
    if seed is None:
        return np.zeros(enhanced_img.shape, bool)
    blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 0)          # Chapter 3: Gaussian smoothing
    grown = region_grow(blurred, seed, TOLERANCE)
    mask = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, SE_OCT_7) > 0
    mask = cv2.morphologyEx(common.keep_liver(mask).astype(np.uint8), cv2.MORPH_CLOSE, SE_OCT_15)
    return add_tumour(enhanced_img, ndi.binary_fill_holes(mask > 0))


def predicted_liver(ct):
    # used by the tumour members when they search inside a predicted (not ground-truth) liver
    return segment_liver(enhance(common.window(ct, 40, 400)))


LIVER_SOURCES = ["ground truth", "predicted liver", "convex hull of predicted liver"]


def search_area(ct, gt_liver, source):
    # where the tumour members look for the tumour: the ground-truth liver, the predicted liver, or the convex hull
    # of the predicted liver (the hull also covers a dark tumour that bites into the liver edge)
    if source == "ground truth" and gt_liver is not None:
        return gt_liver
    liver_mask = predicted_liver(ct)
    if source == "convex hull of predicted liver" and liver_mask.any():
        return convex_hull_image(liver_mask)
    else:
        return liver_mask


def lbp_features(enhanced_img, mask, P=8, R=1):
    # charvi's own features: normalised histogram of uniform LBP codes inside the region
    # (extension of the Chapter 10-11 texture descriptors, not in the slides)
    codes = local_binary_pattern(enhanced_img, P, R, method="uniform")
    hist = np.bincount(codes[mask].astype(int), minlength=P + 2) / mask.sum()
    feats = {}
    for i in range(len(hist)):
        feats[f"lbp_{i}"] = hist[i]
    return feats


def region_features(ct, enhanced_img, mask):
    # shared features first, then LBP (the saved model expects this order)
    feats = common.region_features(ct, enhanced_img, mask)
    extra = lbp_features(enhanced_img, mask)
    for key in extra:
        feats[key] = extra[key]
    return feats


def run(ct, root, gt_liver=None, liver_source="own"):
    bundle = joblib.load(root / MODEL)
    windowed_img = common.window(ct, 40, 400)
    enhanced_img = enhance(windowed_img)
    liver_mask = segment_liver(enhanced_img)
    # tumour candidates: darker than the liver median - 1.5 x std, then k-NN (extension of Chapter 12-13)
    candidates = common.dark_spots(enhanced_img, liver_mask, k=1.5)
    tumour_mask, checked = common.classify_regions(ct, enhanced_img, candidates, bundle, features=region_features)
    result = {}
    result["win"] = windowed_img
    result["enh"] = enhanced_img
    result["liver"] = liver_mask
    result["tumour"] = tumour_mask
    result["note"] = f"{checked} suspicious regions checked by the k-NN, {label(tumour_mask).max()} flagged as tumour"
    return result
