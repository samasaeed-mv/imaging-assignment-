"""Shared helpers used by every member's pipeline: windowing, features, scores, drawing."""
import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops


def window(ct, level=40, width=400):
    # point processing s = T(r): keep the HU window and stretch it to 0-255
    low = level - width / 2
    high = level + width / 2
    img = np.clip(ct, low, high)
    img = (img - low) / (high - low) * 255
    return img.astype(np.uint8)


def keep_liver(mask):
    # connected components -> biggest blob on the image's left side (patient's right)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    best = 0
    best_area = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if centroids[i][0] < 0.6 * mask.shape[1] and area > best_area:
            best = i
            best_area = area
    if best:
        return labels == best
    else:
        return np.zeros(mask.shape, bool)


MIN_AREA = 10
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def shape_features(mask):
    # regional descriptors (area, perimeter, compactness)
    r = regionprops(mask.astype(np.uint8))[0]
    area = r.area
    perim = max(r.perimeter, 1e-6)
    feats = {}
    feats["area"] = area
    feats["perimeter_area_ratio"] = perim / area
    feats["circularity"] = 4 * np.pi * area / perim ** 2
    feats["compactness"] = perim ** 2 / area
    feats["eccentricity"] = r.eccentricity
    feats["solidity"] = r.solidity
    return feats


def glcm_features(enhanced_img, mask, levels=32):
    # GLCM texture inside the region only, averaged over 4 angles
    r = regionprops(mask.astype(np.uint8))[0]
    r0, c0, r1, c1 = r.bbox
    patch = enhanced_img[r0:r1, c0:c1].astype(int)
    m = mask[r0:r1, c0:c1]
    q = np.where(m, 1 + patch * (levels - 1) // 255, 0).astype(np.uint8)   # 1..levels inside, 0 outside
    P = graycomatrix(q, distances=[1], angles=ANGLES, levels=levels + 1)
    P = P[1:, 1:, :, :].astype(float)                                      # drop pairs touching "outside"
    P = P / np.maximum(P.sum(axis=(0, 1), keepdims=True), 1e-12)
    feats = {}
    for prop in ["energy", "contrast", "homogeneity", "correlation"]:
        feats[f"glcm_{prop}"] = np.nan_to_num(graycoprops(P, prop)[0]).mean()
    ent = []
    for k in range(4):
        p_k = P[:, :, 0, k]
        nonzero = p_k[p_k > 0]
        ent.append(-(nonzero * np.log2(nonzero)).sum())
    feats["glcm_entropy"] = np.mean(ent)
    return feats


def first_order_features(ct, mask):
    # 1st-order statistics (statistical moments) of the HU values
    hu = ct[mask].astype(float)
    feats = {}
    feats["mean_HU"] = hu.mean()
    feats["variance_HU"] = hu.var()
    if hu.std() > 0:
        feats["skewness"] = skew(hu)
        feats["kurtosis"] = kurtosis(hu)
    else:
        feats["skewness"] = 0.0
        feats["kurtosis"] = 0.0
    return feats


def dice(pred_mask, gt_mask):
    # Dice = 2|A n B| / (|A| + |B|)
    return 2 * (pred_mask & gt_mask).sum() / (pred_mask.sum() + gt_mask.sum() + 1e-9)


def iou(pred_mask, gt_mask):
    # Jaccard / IoU = |A n B| / |A u B|
    return (pred_mask & gt_mask).sum() / ((pred_mask | gt_mask).sum() + 1e-9)


def overlay(img, pairs, thickness=2):
    # pairs = list of (predicted mask, ground-truth mask). RED = predicted, GREEN = ground truth
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for pred_mask, gt_mask in pairs:
        if gt_mask is not None and gt_mask.any():
            contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, contours, -1, (0, 255, 0), thickness)
        if pred_mask is not None and pred_mask.any():
            contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, contours, -1, (255, 0, 0), thickness)
    return rgb


def region_features(ct, enhanced_img, mask):
    # shape + GLCM + first-order, in this order (the saved models expect this column order)
    feats = {}
    for part in [shape_features(mask), glcm_features(enhanced_img, mask), first_order_features(ct, mask)]:
        for key in part:
            feats[key] = part[key]
    return feats


def classify_regions(ct, enhanced_img, candidates, bundle, min_area=MIN_AREA, features=None):
    # classify every candidate blob; return the tumour blobs and how many were checked
    # `features` = the member's own feature function (default: shape + GLCM + first-order)
    if not features:
        features = region_features
    model = bundle["model"]
    names = bundle["feature_names"]
    tumour_mask = np.zeros(candidates.shape, bool)
    checked = 0
    for r in regionprops(label(candidates)):   # connected components
        if r.area < min_area:
            continue
        m = np.zeros(candidates.shape, bool)
        m[tuple(r.coords.T)] = True
        f = features(ct, enhanced_img, m)
        checked += 1
        row = []
        for n in names:
            row.append(f[n])
        if model.predict(np.array([row]))[0] == 1:
            tumour_mask |= m
    return tumour_mask, checked


def dark_spots(enhanced_img, liver_mask, offset=None, k=None):
    # dark spots inside the liver: darker than median - offset (or median - k x std)
    # erode with a 9x9 square to stay away from the liver edge
    square_9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    inner = cv2.erode(liver_mask.astype(np.uint8), square_9) > 0
    if not inner.any():
        return np.zeros_like(liver_mask)
    if offset is not None:
        drop = offset
    else:
        drop = k * enhanced_img[inner].std()
    limit = np.median(enhanced_img[inner]) - drop   # threshold
    dark = ((enhanced_img < limit) & inner).astype(np.uint8)
    square_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN, square_3) > 0   # opening removes specks
