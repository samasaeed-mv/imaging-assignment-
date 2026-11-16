"""charvi (LIVER, patient 1): window 40/400 -> anisotropic diffusion + CLAHE -> seeded region growing from the liver
core's centroid + dark tumour inside the liver outline (liver) -> dark spots in the liver -> k-NN (tumour regions)."""
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

SE_OCT_15 = np.ones((15, 15), np.uint8)                     # hand-made octagon: square with cut corners
for r in range(15):
    for c in range(15):
        if min(r, 14 - r) + min(c, 14 - c) < 4:
            SE_OCT_15[r, c] = 0
SE_OCT_7 = cv2.resize(SE_OCT_15, (7, 7), interpolation=cv2.INTER_NEAREST)
SE_OCT_61 = cv2.resize(SE_OCT_15, (61, 61), interpolation=cv2.INTER_NEAREST)
LIVER_LOW, LIVER_HIGH, TOLERANCE, FAT_LEVEL = 150, 225, 30, 60


def anisotropic_diffusion(img, iterations=15, kappa=30, gamma=0.15):
    """Perona-Malik diffusion: smooths flat areas but stops at edges."""
    I = img.astype(np.float64)
    for _ in range(iterations):
        diffs = [np.roll(I, -1, 0) - I, np.roll(I, 1, 0) - I, np.roll(I, -1, 1) - I, np.roll(I, 1, 1) - I]
        I = I + gamma * sum(np.exp(-(d / kappa) ** 2) * d for d in diffs)
    return np.clip(I, 0, 255).astype(np.uint8)


def region_grow(img, seed, tolerance, allowed=None):
    """Seeded region growing with a running-average brightness tolerance."""
    h, w = img.shape
    region = np.zeros((h, w), bool); region[seed] = True
    stack, total, count = [seed], float(img[seed]), 1
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not region[yy, xx] and (allowed is None or allowed[yy, xx]):
                if abs(float(img[yy, xx]) - total / count) <= tolerance:
                    region[yy, xx] = True; stack.append((yy, xx)); total += float(img[yy, xx]); count += 1
    return region


def enhance(win):
    return cv2.createCLAHE(2.0, (8, 8)).apply(anisotropic_diffusion(win))


def find_seed(enh):
    """Seed = spatial centroid of the liver core (nearest core pixel if the centroid is outside the core)."""
    band = ((enh >= LIVER_LOW) & (enh <= LIVER_HIGH)).astype(np.uint8)
    core = C.keep_liver(cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))))
    if not core.any():
        return None
    ys, xs = np.nonzero(core)
    i = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
    return ys[i], xs[i]


def add_tumour(enh, liver):
    """Take in the dark tumour region growing leaves out: tumour-grey pixels inside the closed liver outline."""
    outline = ndi.binary_fill_holes(cv2.morphologyEx(liver.astype(np.uint8), cv2.MORPH_CLOSE, SE_OCT_61) > 0)
    dark = outline & ~liver & (enh >= FAT_LEVEL) & (enh < LIVER_LOW)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, SE_OCT_7) > 0
    joined = cv2.morphologyEx((liver | dark).astype(np.uint8), cv2.MORPH_CLOSE, SE_OCT_15) > 0
    return C.keep_liver(ndi.binary_fill_holes(joined))


def segment_liver(enh):
    seed = find_seed(enh)
    if seed is None:
        return np.zeros(enh.shape, bool)
    grown = region_grow(cv2.GaussianBlur(enh, (5, 5), 0), seed, TOLERANCE)
    mask = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, SE_OCT_7) > 0
    mask = cv2.morphologyEx(C.keep_liver(mask).astype(np.uint8), cv2.MORPH_CLOSE, SE_OCT_15)
    return add_tumour(enh, ndi.binary_fill_holes(mask > 0))


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
