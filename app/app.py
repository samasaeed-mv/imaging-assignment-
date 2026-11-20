"""Group prototype: Liver CT -> enhancement -> segmentation -> tumour diagnosis, one pipeline per member.
Run from the project folder:   streamlit run app/app.py

Input: one CT slice OR a group of slices (saved slices, or uploaded .npy in HU / .png windowed 40/400).
Output per slice: liver and tumour masks, RED = predicted outline, GREEN = ground-truth outline, drawn on the
windowed ORIGINAL slice; Dice, IoU and recall per organ. For a group: a per-slice table and slice-level
accuracy, recall (sensitivity), specificity and F1 of the tumour diagnosis."""
import glob
import importlib
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import common                       # shared helpers (window, overlay, dice, iou)

ROOT = Path(__file__).resolve().parent.parent
MEMBERS = {}
MEMBERS["charvi"] = "pipeline_charvi"
MEMBERS["sama"] = "pipeline_sama"
MEMBERS["kritikaa"] = "pipeline_kritikaa"
MEMBERS["evangeline"] = "pipeline_evangeline"

st.set_page_config(page_title="Liver CT Diagnostics", layout="wide")
st.title("Liver CT: segmentation and tumour diagnosis")
st.caption("Classical pipeline, one set of algorithms per group member.  RED = predicted, GREEN = ground truth, "
           "drawn on the windowed original slice.")


# ---------------- helpers ----------------
def png_to_hu(img):
    # a PNG has no HU values. Assume it was saved with the abdomen window 40/400 (0 = -160 HU, 255 = 240 HU)
    # and map it back (the inverse of the Chapter 2.2 point operation). Approximate: everything outside
    # -160..240 HU was clipped when the PNG was made
    return img.astype(np.float32) / 255 * 400 - 160


def read_upload(file):
    # uploaded CT slice -> HU array. .npy = raw HU, .png = abdomen-windowed image converted back (approximate)
    if file.name.lower().endswith(".npy"):
        return np.load(file), False
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return png_to_hu(img), True


def read_mask(file):
    # uploaded PNG mask -> True/False array (or None if there is no file)
    if file is None:
        return None
    return cv2.imdecode(np.frombuffer(file.read(), np.uint8), 0) > 0


def recall(pred_mask, gt_mask):
    # pixel recall (sensitivity): share of the true organ that was found
    if gt_mask.sum():
        return (pred_mask & gt_mask).sum() / gt_mask.sum()
    else:
        return np.nan


def metric_or_na(col, name, value):
    # show a number with 3 decimals, or "n/a" when there is no value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        col.metric(name, "n/a")
    else:
        col.metric(name, f"{value:.3f}")


def organ_scores(pred_mask, gt_mask):
    scores = {}
    if gt_mask is None or (not pred_mask.any() and not gt_mask.any()):
        # no ground truth, or nothing to find and nothing found
        scores["Dice"] = np.nan
        scores["IoU"] = np.nan
        scores["recall"] = np.nan
    else:
        scores["Dice"] = common.dice(pred_mask, gt_mask)
        scores["IoU"] = common.iou(pred_mask, gt_mask)
        scores["recall"] = recall(pred_mask, gt_mask)
    return scores


# ---------------- sidebar: member ----------------
st.sidebar.header("1. Choose a member's pipeline")
member = st.sidebar.selectbox("Member", list(MEMBERS))
M = importlib.import_module(MEMBERS[member])
st.sidebar.caption(f"**{M.ORGAN.upper()} member, patient {M.PATIENT}**  \n{M.ALGORITHMS}")

liver_source = "own"
if M.ORGAN == "tumour":
    import pipeline_charvi                # owns the predicted liver and the search-area options
    sources = pipeline_charvi.LIVER_SOURCES
    liver_source = st.sidebar.radio("Liver used to search for the tumour", sources,
                                    index=sources.index(M.DEFAULT_LIVER_SOURCE))
    st.sidebar.caption(f"Default for {member}: **{M.DEFAULT_LIVER_SOURCE}** (charvi's region growing), the better of "
                       "the two automatic search areas in Phase 2. 'ground truth' = upper bound with a perfect liver.")

# ---------------- sidebar: input (one slice or a group) ----------------
st.sidebar.header("2. Choose CT slice(s)")
mode = st.sidebar.radio("Mode", ["One slice", "Group of slices"])
source = st.sidebar.radio("Source", ["Saved slices", "Upload"])
items = []                                       # list of (name, ct in HU, gt liver or None, gt tumour or None, from_png)
if source == "Saved slices":
    folders = []
    for p in (ROOT / "data" / "slices").iterdir():
        if p.is_dir():
            folders.append(p.name)
    folders = sorted(folders)
    if member in folders:
        default_folder = folders.index(member)
    else:
        default_folder = 0
    folder = st.sidebar.selectbox("Slices folder", folders, index=default_folder)

    names = []
    for ct_file in sorted(glob.glob(str(ROOT / "data" / "slices" / folder / "*_ct.npy"))):
        names.append(Path(ct_file).name.replace("_ct.npy", ""))

    if mode == "One slice":
        if names:
            chosen = [st.sidebar.selectbox("Slice", names)]
        else:
            chosen = []
    else:
        chosen = st.sidebar.multiselect("Slices (empty = all in the folder)", names)
        if not chosen:
            chosen = names
    for n in chosen:
        base = ROOT / "data" / "slices" / folder / n
        ct = np.load(f"{base}_ct.npy")
        gt_liver = cv2.imread(f"{base}_liver.png", 0) > 0
        gt_tumour = cv2.imread(f"{base}_tumour.png", 0) > 0
        items.append((n, ct, gt_liver, gt_tumour, False))
else:
    st.sidebar.caption("CT: **.npy** (raw HU) or **.png** (saved with the abdomen window 40/400). Optional ground truth: "
                       "PNG masks named `<slice>_liver.png` / `<slice>_tumour.png` (uploaded together with the CT).")
    files = st.sidebar.file_uploader("CT slice(s) + optional masks", type=["npy", "png"],
                                     accept_multiple_files=(mode == "Group of slices"))
    if files is None:
        files = []
    elif not isinstance(files, list):
        files = [files]

    # the mask files, by name without ".png"
    masks = {}
    for f in files:
        if f.name.lower().endswith(("_liver.png", "_tumour.png")):
            masks[f.name[:-4]] = f
    for f in files:
        if f.name[:-4] in masks:
            continue                                    # this file is a mask, not a CT slice
        stem = f.name.rsplit(".", 1)[0]
        if stem.endswith("_ct"):
            base = stem[:-3]
        else:
            base = stem
        ct, from_png = read_upload(f)
        gt_liver = read_mask(masks.get(f"{base}_liver"))
        gt_tumour = read_mask(masks.get(f"{base}_tumour"))
        items.append((base, ct, gt_liver, gt_tumour, from_png))

if not items:
    st.info("Choose or upload CT slice(s) in the sidebar to start.")
    st.stop()

any_png = False
any_missing_liver = False
for item in items:
    if item[4]:
        any_png = True
    if item[2] is None:
        any_missing_liver = True
if any_png:
    st.warning("PNG input: HU values are reconstructed from the abdomen window 40/400, so tissue outside -160..240 HU "
               "is clipped and the narrow tumour windows see a coarser image than with .npy input.")
if M.ORGAN == "tumour" and liver_source == "ground truth" and any_missing_liver:
    st.warning("Some slices have no ground-truth liver, so the predicted liver is used for them instead.")

# ---------------- run ----------------
results = []
progress = st.progress(0.0, text="Running the pipeline...")
for k in range(len(items)):
    name, ct, gt_liver, gt_tumour, _ = items[k]
    # without a ground-truth liver, "ground truth" falls back to the predicted liver
    if liver_source == "ground truth" and gt_liver is None:
        src = "predicted liver"
    else:
        src = liver_source
    results.append(M.run(ct, ROOT, gt_liver=gt_liver, liver_source=src))
    progress.progress((k + 1) / len(items), text=f"Running the pipeline... {k + 1}/{len(items)}")
progress.empty()

if liver_source == "ground truth":
    liver_label = "Liver (ground truth, used as search area)"
elif liver_source == "convex hull of predicted liver":
    liver_label = "Convex hull of the predicted liver (search area)"
else:
    liver_label = "Predicted liver"


def show_slice(name, R, gt_liver, gt_tumour):
    # Phase 1 images, diagnosis, and the per-organ overlays + scores for one slice
    st.subheader(f"Slice {name}")
    c1, c2 = st.columns(2)
    c1.image(R["win"], caption="Windowed original")
    c2.image(R["enh"], caption="Enhanced (this member's Phase 1)")
    if R["tumour"].any():
        diagnosis = "TUMOUR DETECTED"
    else:
        diagnosis = "No tumour detected"
    st.markdown(f"**Diagnosis for this slice: {diagnosis}**  ({R['note']})")
    tab_liver, tab_tumour, tab_all = st.tabs(["Liver", "Tumour", "All organs together"])

    # one tab per organ: overlay on the left, predicted mask + scores on the right
    organ_tabs = [tab_liver, tab_tumour]
    organ_masks = [R["liver"], R["tumour"]]
    organ_gts = [gt_liver, gt_tumour]
    organ_captions = [liver_label, "Predicted tumour mask"]
    for i in range(2):
        with organ_tabs[i]:
            a, b = st.columns([2, 1])
            a.image(common.overlay(R["win"], [(organ_masks[i], organ_gts[i])]))
            b.image(organ_masks[i].astype(np.uint8) * 255, caption=organ_captions[i])
            s = organ_scores(organ_masks[i], organ_gts[i])
            m1, m2, m3 = b.columns(3)
            metric_or_na(m1, "Dice", s["Dice"])
            metric_or_na(m2, "IoU", s["IoU"])
            metric_or_na(m3, "Recall", s["recall"])
    with tab_all:
        st.image(common.overlay(R["win"], [(R["liver"], gt_liver), (R["tumour"], gt_tumour)]))


if mode == "One slice":
    name, _, gt_liver, gt_tumour, _ = items[0]
    show_slice(name, results[0], gt_liver, gt_tumour)
else:
    # ---------------- group: per-slice table + slice-level diagnosis scores ----------------
    rows = []
    for k in range(len(items)):
        name, _, gt_liver, gt_tumour, _ = items[k]
        R = results[k]
        lv = organ_scores(R["liver"], gt_liver)
        tm = organ_scores(R["tumour"], gt_tumour)
        row = {}
        row["slice"] = name
        row["liver Dice"] = lv["Dice"]
        row["liver IoU"] = lv["IoU"]
        row["tumour Dice"] = tm["Dice"]
        row["tumour IoU"] = tm["IoU"]
        row["tumour recall"] = tm["recall"]
        if gt_tumour is None:
            row["true tumour on slice"] = None
        else:
            row["true tumour on slice"] = bool(gt_tumour.any())
        row["predicted tumour"] = bool(R["tumour"].any())
        rows.append(row)
    table = pd.DataFrame(rows)
    st.subheader(f"Group of {len(items)} slices: per-slice results")
    st.dataframe(table.round(3), hide_index=True)

    known = table.dropna(subset=["true tumour on slice"])
    st.subheader("Slice-level tumour diagnosis (slices with ground truth)")
    if len(known):
        truth = known["true tumour on slice"].astype(bool)
        pred = known["predicted tumour"].astype(bool)
        tp = int((truth & pred).sum())
        tn = int((~truth & ~pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        if tp + fn:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = np.nan
        if tn + fp:
            specificity = tn / (tn + fp)
        else:
            specificity = np.nan
        if tp + fp + fn:
            f1 = 2 * tp / (2 * tp + fp + fn)
        else:
            f1 = np.nan
        cols = st.columns(6)
        metric_or_na(cols[0], "Accuracy", (tp + tn) / len(known))
        metric_or_na(cols[1], "Recall (sensitivity)", sensitivity)
        metric_or_na(cols[2], "Specificity", specificity)
        metric_or_na(cols[3], "F1", f1)
        metric_or_na(cols[4], "Mean liver Dice", table["liver Dice"].mean())
        metric_or_na(cols[5], "Mean tumour Dice", table["tumour Dice"].mean())
        st.caption(f"TP {tp}, FN {fn}, TN {tn}, FP {fp}. Tumour-free slices for testing specificity are in the "
                   f"`<member>_healthy` folders. Tumour Dice is averaged over slices that have a true or predicted tumour.")
    else:
        st.info("No ground truth for these slices, so no scores can be computed.")

    pick = st.selectbox("Show one slice of the group in detail", table["slice"])
    k = list(table["slice"]).index(pick)
    show_slice(pick, results[k], items[k][2], items[k][3])


# ---------------- results tables ----------------
def member_results(mod):
    # mean Dice of the member's final segmentation method (Phase 2) and the Phase 3 test scores
    seg_file, seg_method = mod.PHASE2_METRICS
    seg_dice = None
    clf = None
    if (ROOT / seg_file).exists():
        seg = pd.read_csv(ROOT / seg_file)
        seg_dice = seg[seg.method == seg_method]["Dice"].mean()
    if (ROOT / mod.PHASE3_METRICS).exists():
        clf = pd.read_csv(ROOT / mod.PHASE3_METRICS).iloc[0]
    return seg_dice, clf


st.subheader(f"Overall results for {member} (Phase 2 test slices, Phase 3 grouped 30% test set)")
seg_dice, clf = member_results(M)
cols = st.columns(5)
metric_or_na(cols[0], f"{M.ORGAN.capitalize()} Dice (mean)", seg_dice)
score_names = ["Accuracy", "F1", "Recall (sensitivity)", "Specificity"]
score_keys = ["accuracy", "F1", "sensitivity (recall)", "specificity"]
for i in range(4):
    if clf is None:
        value = None
    else:
        value = clf[score_keys[i]]
    metric_or_na(cols[i + 1], score_names[i], value)

st.subheader("Group overview")
rows = []
for name in MEMBERS:
    mod = importlib.import_module(MEMBERS[name])
    d, c = member_results(mod)
    row = {}
    row["member"] = name
    row["organ"] = mod.ORGAN
    row["patient"] = mod.PATIENT
    row["segmentation Dice"] = d
    if c is None:
        row["accuracy"] = None
        row["F1"] = None
        row["recall"] = None
        row["specificity"] = None
    else:
        row["accuracy"] = c["accuracy"]
        row["F1"] = c["F1"]
        row["recall"] = c["sensitivity (recall)"]
        row["specificity"] = c["specificity"]
    rows.append(row)
st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
