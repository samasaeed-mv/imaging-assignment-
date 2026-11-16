"""evangeline (TUMOUR, patient 29): narrow window 90/150 -> anisotropic diffusion (kappa 60) -> marker-controlled
watershed inside the liver (tumour) -> decision tree on SLIC superpixels (diagnosis)."""
import numpy as np
import cv2
import joblib
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed, slic
from skimage.measure import regionprops, label
import common as C
import pipeline_charvi

NAME, ORGAN, PATIENT = "evangeline", "tumour", 29
ALGORITHMS = ("Window 90/150 -> anisotropic diffusion (kappa 60) | Tumour: adaptive threshold vs marker-controlled "
              "watershed (used) | Diagnosis: decision tree on SLIC superpixels")
PHASE2_METRICS = ("phase2/outputs_evangeline/phase2_metrics.csv", "watershed")
PHASE3_METRICS = "phase3/outputs_evangeline/phase3_metrics.csv"
MODEL = "phase3/outputs_evangeline/tree_model.joblib"
SE_ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def enhance(win):
    return pipeline_charvi.anisotropic_diffusion(win, kappa=60)   # temporary: replaced by evangeline's own filter next

def clean(mask, min_area=30):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, SE_ELLIPSE_5) > 0
    lab = label(mask); out = np.zeros(mask.shape, bool)
    for r in regionprops(lab):
        if r.area >= min_area:
            out[lab == r.label] = True
    return ndi.binary_fill_holes(cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, SE_ELLIPSE_5) > 0)

def segment_tumour(enh, liver, k=0.75):
    roi = cv2.erode(liver.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if roi.sum() < 50:
        return np.zeros(enh.shape, bool)
    smooth = cv2.GaussianBlur(enh, (5, 5), 0)
    med, sd = np.median(smooth[roi]), smooth[roi].std()
    markers = np.zeros(enh.shape, np.int32)
    markers[~cv2.dilate(liver.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 1
    markers[(smooth > med) & roi] = 1
    markers[cv2.morphologyEx(((smooth < med - k * sd) & roi).astype(np.uint8), cv2.MORPH_OPEN, SE_ELLIPSE_5) > 0] = 2
    return clean((watershed(sobel(smooth / 255.0), markers) == 2) & roi)

def diagnose(ct, enh, liver, bundle):
    """Cut the liver into ~20 superpixels and let the decision tree label each one."""
    model, names = bundle["model"], bundle["feature_names"]
    if liver.sum() < 50:
        return 0, 0
    sp = slic(enh, n_segments=20, compactness=0.1, mask=liver, channel_axis=None, start_label=1)
    checked = flagged = 0
    for i in range(1, sp.max() + 1):
        m = sp == i
        if m.sum() < 10:
            continue
        f = C.region_features(ct, enh, m)
        checked += 1
        flagged += int(model.predict(np.array([[f[n] for n in names]]))[0] == 1)
    return checked, flagged

def run(ct, root, gt_liver=None, liver_source="ground truth"):
    bundle = joblib.load(root / MODEL)
    win = C.window(ct, 90, 150)
    enh = enhance(win)
    liver = gt_liver if (liver_source == "ground truth" and gt_liver is not None) else pipeline_charvi.predicted_liver(ct)
    tumour = segment_tumour(enh, liver)
    checked, flagged = diagnose(ct, enh, liver, bundle)
    if flagged == 0:
        tumour = np.zeros_like(tumour)          # the classifier overrules the segmentation: no tumour
    return {"win": win, "enh": enh, "liver": liver, "tumour": tumour,
            "note": f"{checked} liver superpixels checked by the decision tree, {flagged} flagged as tumour"}
