"""Group prototype: Liver CT -> enhancement -> segmentation -> tumour diagnosis, one pipeline per member.
Run from the project folder:   streamlit run app/app.py"""
import glob
import importlib
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import common as P                       # shared helpers (overlay, dice, iou)

ROOT = Path(__file__).resolve().parent.parent
MEMBERS = {"charvi": "pipeline_charvi", "sama": "pipeline_sama",
           "kritikaa": "pipeline_kritikaa", "evangeline": "pipeline_evangeline"}

st.set_page_config(page_title="Liver CT Diagnostics", layout="wide")
st.title("Liver CT: segmentation and tumour diagnosis")
st.caption("Classical pipeline, one set of algorithms per group member.  RED = predicted, GREEN = ground truth.")

def read_mask(file):
    return None if file is None else cv2.imdecode(np.frombuffer(file.read(), np.uint8), 0) > 0

def metric_or_na(col, name, value):
    col.metric(name, "n/a" if value is None or np.isnan(value) else f"{value:.3f}")

# ---------------- sidebar ----------------
st.sidebar.header("1. Choose a member's pipeline")
member = st.sidebar.selectbox("Member", list(MEMBERS))
M = importlib.import_module(MEMBERS[member])
st.sidebar.caption(f"**{M.ORGAN.upper()} member, patient {M.PATIENT}**  \n{M.ALGORITHMS}")

liver_source = "own"
if M.ORGAN == "tumour":
    liver_source = st.sidebar.radio("Liver used to search for the tumour",
                                    ["ground truth", "predicted (charvi region growing)"])

st.sidebar.header("2. Choose a CT slice")
source = st.sidebar.radio("Source", ["Saved slices", "Upload a slice"])
ct = gt_liver = gt_tumour = None
if source == "Saved slices":
    folders = sorted(p.name for p in (ROOT / "data" / "slices").iterdir() if p.is_dir())
    folder = st.sidebar.selectbox("Slices folder", folders, index=folders.index(member) if member in folders else 0)
    cts = sorted(glob.glob(str(ROOT / "data" / "slices" / folder / "*_ct.npy")))
    if cts:
        chosen = st.sidebar.selectbox("Slice", [Path(c).name.replace("_ct.npy", "") for c in cts])
        base = ROOT / "data" / "slices" / folder / chosen
        ct = np.load(f"{base}_ct.npy")
        gt_liver = cv2.imread(f"{base}_liver.png", 0) > 0
        gt_tumour = cv2.imread(f"{base}_tumour.png", 0) > 0
else:
    ct_file = st.sidebar.file_uploader("CT slice (.npy, raw HU)", type="npy")
    liver_file = st.sidebar.file_uploader("Ground-truth liver mask (.png, optional)", type="png")
    tumour_file = st.sidebar.file_uploader("Ground-truth tumour mask (.png, optional)", type="png")
    if ct_file is not None:
        ct = np.load(ct_file)
        gt_liver, gt_tumour = read_mask(liver_file), read_mask(tumour_file)

if ct is None:
    st.info("Choose or upload a CT slice in the sidebar to start.")
    st.stop()
if M.ORGAN == "tumour" and liver_source == "ground truth" and gt_liver is None:
    st.warning("No ground-truth liver for this slice, so the predicted liver is used instead.")

# ---------------- run ----------------
with st.spinner("Running the pipeline..."):
    R = M.run(ct, ROOT, gt_liver=gt_liver, liver_source=liver_source)

st.subheader("Phase 1: enhancement")
c1, c2 = st.columns(2)
c1.image(R["win"], caption="Windowed")
c2.image(R["enh"], caption="Enhanced (this member's Phase 1)")

st.subheader("Phase 2 and 3: segmentation and diagnosis")
diagnosis = "TUMOUR DETECTED" if R["tumour"].any() else "No tumour detected"
st.markdown(f"**Diagnosis for this slice: {diagnosis}**  ({R['note']})")

liver_label = "Predicted liver" if (M.ORGAN == "liver" or liver_source != "ground truth") else "Liver (ground truth, used as search area)"
tab_liver, tab_tumour, tab_all = st.tabs(["Liver", "Tumour", "All organs together"])
with tab_liver:
    a, b = st.columns([2, 1])
    a.image(P.overlay(R["enh"], [(R["liver"], gt_liver)]))
    b.image(R["liver"].astype(np.uint8) * 255, caption=liver_label)
    if gt_liver is not None:
        m1, m2 = b.columns(2)
        m1.metric("Dice", f"{P.dice(R['liver'], gt_liver):.3f}")
        m2.metric("IoU", f"{P.iou(R['liver'], gt_liver):.3f}")
with tab_tumour:
    a, b = st.columns([2, 1])
    a.image(P.overlay(R["enh"], [(R["tumour"], gt_tumour)]))
    b.image(R["tumour"].astype(np.uint8) * 255, caption="Predicted tumour mask")
    if gt_tumour is not None:
        m1, m2 = b.columns(2)
        m1.metric("Dice", f"{P.dice(R['tumour'], gt_tumour):.3f}" if (R["tumour"].any() or gt_tumour.any()) else "n/a")
        m2.metric("Tumour pixels (pred / true)", f"{int(R['tumour'].sum())} / {int(gt_tumour.sum())}")
with tab_all:
    st.image(P.overlay(R["enh"], [(R["liver"], gt_liver), (R["tumour"], gt_tumour)]))

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

st.subheader(f"Overall results for {member} (all test slices)")
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
                 "F1": None if c is None else c["F1"], "recall": None if c is None else c["sensitivity (recall)"]})
st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
