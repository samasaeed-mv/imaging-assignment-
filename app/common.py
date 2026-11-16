"""Shared helpers used by every member's pipeline: windowing, features, scores, drawing."""
import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops

def window(ct, level=40, width=400):
    """HU -> 0-255 using the abdomen soft-tissue window."""
    lo, hi = level - width / 2, level + width / 2
    return ((np.clip(ct, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def keep_liver(mask):
    """Biggest blob on the image's left side (patient's right)."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    best, best_area = 0, 0
    for i in range(1, n):
        if centroids[i][0] < 0.6 * mask.shape[1] and stats[i, cv2.CC_STAT_AREA] > best_area:
            best, best_area = i, stats[i, cv2.CC_STAT_AREA]
    return labels == best if best else np.zeros(mask.shape, bool)

MIN_AREA = 10
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

def shape_features(mask):
    r = regionprops(mask.astype(np.uint8))[0]
    area, perim = r.area, max(r.perimeter, 1e-6)
    return {"area": area, "perimeter_area_ratio": perim / area,
            "circularity": 4 * np.pi * area / perim ** 2, "compactness": perim ** 2 / area,
            "eccentricity": r.eccentricity, "solidity": r.solidity}

def glcm_features(enh, mask, levels=32):
    r = regionprops(mask.astype(np.uint8))[0]
    r0, c0, r1, c1 = r.bbox
    patch, m = enh[r0:r1, c0:c1].astype(int), mask[r0:r1, c0:c1]
    q = np.where(m, 1 + patch * (levels - 1) // 255, 0).astype(np.uint8)
    P = graycomatrix(q, distances=[1], angles=ANGLES, levels=levels + 1)[1:, 1:, :, :].astype(float)
    P = P / np.maximum(P.sum(axis=(0, 1), keepdims=True), 1e-12)
    feats = {f"glcm_{p}": np.nan_to_num(graycoprops(P, p)[0]).mean()
             for p in ["energy", "contrast", "homogeneity", "correlation"]}
    ent = [-(P[:, :, 0, k][P[:, :, 0, k] > 0] * np.log2(P[:, :, 0, k][P[:, :, 0, k] > 0])).sum() for k in range(4)]
    feats["glcm_entropy"] = np.mean(ent)
    return feats

def first_order_features(ct, mask):
    hu = ct[mask].astype(float)
    return {"mean_HU": hu.mean(), "variance_HU": hu.var(),
            "skewness": skew(hu) if hu.std() > 0 else 0.0,
            "kurtosis": kurtosis(hu) if hu.std() > 0 else 0.0}

def dice(pred, gt):
    return 2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-9)

def iou(pred, gt):
    return (pred & gt).sum() / ((pred | gt).sum() + 1e-9)

def overlay(img, pairs, thickness=2):
    """pairs = list of (predicted mask, ground-truth mask). RED = predicted, GREEN = ground truth."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for pred, gt in pairs:
        for mask, colour in [(gt, (0, 255, 0)), (pred, (255, 0, 0))]:
            if mask is not None and mask.any():
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, contours, -1, colour, thickness)
    return rgb

def region_features(ct, enh, mask):
    return {**shape_features(mask), **glcm_features(enh, mask), **first_order_features(ct, mask)}

def classify_regions(ct, enh, candidates, bundle, min_area=MIN_AREA, features=None):
    """Run a trained classifier on every candidate blob; return the blobs it calls tumour + how many were checked.
    `features` = the member's own feature function (default: shape + GLCM + first-order)."""
    features = features or region_features
    model, names = bundle["model"], bundle["feature_names"]
    out, checked = np.zeros(candidates.shape, bool), 0
    for r in regionprops(label(candidates)):
        if r.area < min_area:
            continue
        m = np.zeros(candidates.shape, bool); m[tuple(r.coords.T)] = True
        f = features(ct, enh, m)
        checked += 1
        if model.predict(np.array([[f[n] for n in names]]))[0] == 1:
            out |= m
    return out, checked

def dark_spots(enh, liver, offset=None, k=None):
    """Dark spots inside the liver: darker than median - offset (or median - k x std)."""
    inner = cv2.erode(liver.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if not inner.any():
        return np.zeros_like(liver)
    limit = np.median(enh[inner]) - (offset if offset is not None else k * enh[inner].std())
    return cv2.morphologyEx(((enh < limit) & inner).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
