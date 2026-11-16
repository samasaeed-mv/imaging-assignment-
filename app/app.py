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
import common as P                       # shared helpers (window, overlay, dice, iou)

ROOT = Path(__file__).resolve().parent.parent
MEMBERS = {"charvi": "pipeline_charvi", "sama": "pipeline_sama",
           "kritikaa": "pipeline_kritikaa", "evangeline": "pipeline_evangeline"}

st.set_page_config(page_title="Liver CT Diagnostics", layout="wide")
st.title("Liver CT: segmentation and tumour diagnosis")
st.caption("Classical pipeline, one set of algorithms per group member.  RED = predicted, GREEN = ground truth, "
           "drawn on the windowed original slice.")


# ---------------- helpers ----------------
def png_to_hu(img):
    """A PNG has no HU values. Assume it was saved with the abdomen window 40/400 (0 = -160 HU, 255 = 240 HU)
    and map it back. Approximate: everything outside -160..240 HU was clipped when the PNG was made."""
    return img.astype(np.float32) / 255 * 400 - 160


def read_upload(file):
    """Uploaded CT slice -> HU array. .npy = raw HU, .png = abdomen-windowed image converted back (approximate)."""
    if file.name.lower().endswith(".npy"):
        return np.load(file), False
    return png_to_hu(cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)), True


def read_mask(file):
    return None if file is None else cv2.imdecode(np.frombuffer(file.read(), np.uint8), 0) > 0


def recall(pred, gt):
    """Pixel recall (sensitivity): share of the true organ that was found."""
    return (pred & gt).sum() / gt.sum() if gt.sum() else np.nan


def metric_or_na(col, name, value):
    col.metric(name, "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.3f}")


def organ_scores(pred, gt):
    if gt is None:
        return {"Dice": np.nan, "IoU": np.nan, "recall": np.nan}
    if not pred.any() and not gt.any():
        return {"Dice": np.nan, "IoU": np.nan, "recall": np.nan}       # nothing to find and nothing found
    return {"Dice": P.dice(pred, gt), "IoU": P.iou(pred, gt), "recall": recall(pred, gt)}


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
    folders = sorted(p.name for p in (ROOT / "data" / "slices").iterdir() if p.is_dir())
    folder = st.sidebar.selectbox("Slices folder", folders, index=folders.index(member) if member in folders else 0)
    names = [Path(c).name.replace("_ct.npy", "") for c in sorted(glob.glob(str(ROOT / "data" / "slices" / folder / "*_ct.npy")))]
    if mode == "One slice":
        chosen = [st.sidebar.selectbox("Slice", names)] if names else []
    else:
        chosen = st.sidebar.multiselect("Slices (empty = all in the folder)", names) or names
    for n in chosen:
        base = ROOT / "data" / "slices" / folder / n
        items.append((n, np.load(f"{base}_ct.npy"), cv2.imread(f"{base}_liver.png", 0) > 0,
                      cv2.imread(f"{base}_tumour.png", 0) > 0, False))
else:
    st.sidebar.caption("CT: **.npy** (raw HU) or **.png** (saved with the abdomen window 40/400). Optional ground truth: "
                       "PNG masks named `<slice>_liver.png` / `<slice>_tumour.png` (uploaded together with the CT).")
    files = st.sidebar.file_uploader("CT slice(s) + optional masks", type=["npy", "png"],
                                     accept_multiple_files=(mode == "Group of slices"))
    files = [] if files is None else (files if isinstance(files, list) else [files])
    masks = {f.name[:-4]: f for f in files if f.name.lower().endswith(("_liver.png", "_tumour.png"))}
    for f in files:
        if f.name[:-4] in masks:
            continue                                    # this file is a mask, not a CT slice
        stem = f.name.rsplit(".", 1)[0]
        base = stem[:-3] if stem.endswith("_ct") else stem
        ct, from_png = read_upload(f)
        items.append((base, ct, read_mask(masks.get(f"{base}_liver")), read_mask(masks.get(f"{base}_tumour")), from_png))

if not items:
    st.info("Choose or upload CT slice(s) in the sidebar to start.")
    st.stop()
if any(it[4] for it in items):
    st.warning("PNG input: HU values are reconstructed from the abdomen window 40/400, so tissue outside -160..240 HU "
               "is clipped and the narrow tumour windows see a coarser image than with .npy input.")
if M.ORGAN == "tumour" and liver_source == "ground truth" and any(it[2] is None for it in items):
    st.warning("Some slices have no ground-truth liver, so the predicted liver is used for them instead.")

# ---------------- run ----------------
results = []
progress = st.progress(0.0, text="Running the pipeline...")
for k, (name, ct, gt_liver, gt_tumour, _) in enumerate(items):
    src = liver_source if (liver_source != "ground truth" or gt_liver is not None) else "predicted liver"
    results.append(M.run(ct, ROOT, gt_liver=gt_liver, liver_source=src))
    progress.progress((k + 1) / len(items), text=f"Running the pipeline... {k + 1}/{len(items)}")
progress.empty()

liver_label = {"ground truth": "Liver (ground truth, used as search area)",
               "convex hull of predicted liver": "Convex hull of the predicted liver (search area)"}.get(liver_source, "Predicted liver")


def show_slice(name, R, gt_liver, gt_tumour):
    """Phase 1 images, diagnosis, and the per-organ overlays + scores for one slice."""
    st.subheader(f"Slice {name}")
    c1, c2 = st.columns(2)
    c1.image(R["win"], caption="Windowed original")
    c2.image(R["enh"], caption="Enhanced (this member's Phase 1)")
    diagnosis = "TUMOUR DETECTED" if R["tumour"].any() else "No tumour detected"
    st.markdown(f"**Diagnosis for this slice: {diagnosis}**  ({R['note']})")
    tab_liver, tab_tumour, tab_all = st.tabs(["Liver", "Tumour", "All organs together"])
    for tab, mask, gt, caption in [(tab_liver, R["liver"], gt_liver, liver_label),
                                   (tab_tumour, R["tumour"], gt_tumour, "Predicted tumour mask")]:
        with tab:
            a, b = st.columns([2, 1])
            a.image(P.overlay(R["win"], [(mask, gt)]))
            b.image(mask.astype(np.uint8) * 255, caption=caption)
            s = organ_scores(mask, gt)
            m1, m2, m3 = b.columns(3)
            metric_or_na(m1, "Dice", s["Dice"]); metric_or_na(m2, "IoU", s["IoU"]); metric_or_na(m3, "Recall", s["recall"])
    with tab_all:
        st.image(P.overlay(R["win"], [(R["liver"], gt_liver), (R["tumour"], gt_tumour)]))


if mode == "One slice":
    name, _, gt_liver, gt_tumour, _ = items[0]
    show_slice(name, results[0], gt_liver, gt_tumour)
else:
    # ---------------- group: per-slice table + slice-level diagnosis scores ----------------
    rows = []
    for (name, _, gt_liver, gt_tumour, _), R in zip(items, results):
        lv, tm = organ_scores(R["liver"], gt_liver), organ_scores(R["tumour"], gt_tumour)
        rows.append({"slice": name, "liver Dice": lv["Dice"], "liver IoU": lv["IoU"],
                     "tumour Dice": tm["Dice"], "tumour IoU": tm["IoU"], "tumour recall": tm["recall"],
                     "true tumour on slice": None if gt_tumour is None else bool(gt_tumour.any()),
                     "predicted tumour": bool(R["tumour"].any())})
    table = pd.DataFrame(rows)
    st.subheader(f"Group of {len(items)} slices: per-slice results")
    st.dataframe(table.round(3), hide_index=True)

    known = table.dropna(subset=["true tumour on slice"])
    st.subheader("Slice-level tumour diagnosis (slices with ground truth)")
    if len(known):
        truth, pred = known["true tumour on slice"].astype(bool), known["predicted tumour"].astype(bool)
        tp, tn = int((truth & pred).sum()), int((~truth & ~pred).sum())
        fp, fn = int((~truth & pred).sum()), int((truth & ~pred).sum())
        cols = st.columns(6)
        metric_or_na(cols[0], "Accuracy", (tp + tn) / len(known))
        metric_or_na(cols[1], "Recall (sensitivity)", tp / (tp + fn) if tp + fn else np.nan)
        metric_or_na(cols[2], "Specificity", tn / (tn + fp) if tn + fp else np.nan)
        metric_or_na(cols[3], "F1", 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan)
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
    seg_file, seg_method = mod.PHASE2_METRICS
    seg_dice = clf = None
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
for col, (name, key) in zip(cols[1:], [("Accuracy", "accuracy"), ("F1", "F1"),
                                       ("Recall (sensitivity)", "sensitivity (recall)"), ("Specificity", "specificity")]):
    metric_or_na(col, name, None if clf is None else clf[key])

st.subheader("Group overview")
rows = []
for name, modname in MEMBERS.items():
    mod = importlib.import_module(modname)
    d, c = member_results(mod)
    rows.append({"member": name, "organ": mod.ORGAN, "patient": mod.PATIENT,
                 "segmentation Dice": d, "accuracy": None if c is None else c["accuracy"],
                 "F1": None if c is None else c["F1"], "recall": None if c is None else c["sensitivity (recall)"],
                 "specificity": None if c is None else c["specificity"]})
st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
