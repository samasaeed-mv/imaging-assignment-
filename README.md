# Automated Abdominal CT Organ & Lesion Diagnostics Pipeline (LiTS, liver + liver tumour)

Classical image processing only (no deep learning / TensorFlow / pre-trained models).
Raw CT slice -> Phase 1 enhancement -> Phase 2 segmentation -> Phase 3 features + classification -> diagnosis (healthy vs tumour).

## Who did what

| | charvi | sama | kritikaa | evangeline |
|---|---|---|---|---|
| Organ | liver | liver | tumour | tumour |
| Patient | 1 | 0 | 22 | 29 |
| Window (level / width) | 40 / 400 | 40 / 400 | 80 / 200 | 90 / 150 |
| Spatial filter | **anisotropic diffusion** vs Gaussian | **bilateral** vs median | **non-local means** vs bilateral, Gaussian | **guided filter** vs anisotropic diffusion (8 neighbours), median |
| Contrast | **CLAHE** vs HE | **CLAHE** vs HE | **CLAHE** vs HE | none (CLAHE and HE tested, both lowered the tumour CNR) |
| Frequency filter (compared only) | homomorphic | Butterworth lowpass (D0 60, n 2) | Butterworth lowpass (D0 50, n 3) vs Gaussian lowpass (D0 50) | homomorphic vs notch-reject |
| Segmentation | adaptive threshold vs **seeded region growing** | Multi-Otsu vs **marker-controlled watershed** | Niblack vs statistical region growing vs **geodesic active contour** | **Sauvola** vs distance-transform watershed vs Chan-Vese |
| Custom structuring element | cross 9x9 + hand-made octagon (7, 15, 61) | hand-made diamond (7, 15) + ellipse 25 | hand-made plus (5, 7) | hand-made hexagon (5, 9) |
| Own feature set | LBP histogram | Hu moments | border sharpness | Laws texture energy |
| Classifier | k-NN | SVM (RBF) | random forest | decision tree on SLIC superpixels |

Bold = the method used in the final pipeline and the app. All members also use shape, GLCM and first-order HU features. The frequency filters are compared in Phase 1 but not used in any final pipeline. The tumour members search inside charvi's predicted liver (kritikaa: its convex hull). Every file, folder, figure and failed case carries the member's name.

## Folders

| Folder | What is in it |
|---|---|
| `data/raw_volumes/` | LiTS `volume-N.nii.gz` + `segmentation-N.nii.gz` (gzipped originals) |
| `data/slices/<name>/` | each member's 2D test slices: `*_ct.npy` (raw HU), `*_liver.png`, `*_tumour.png` |
| `data/slices/<name>_healthy/` | tumour-free slices of the same patient, only for the false-alarm test |
| `data/data_log.csv` | which patient and slices each member used |
| `phase1/` | `01_slicing_<name>`, `02_enhancement_<name>` notebooks + `outputs_<name>/` (enhanced slices, PSNR/CNR table, noise test) |
| `phase2/` | `03_segmentation_<name>` notebooks + `outputs_<name>/` (predicted masks per method, Dice/IoU table, sensitivity table; tumour members also the predicted-liver and convex-hull results) |
| `phase3/` | `04_classification_<name>` notebooks + `outputs_<name>/` (features, metrics, feature tests, ablation, trained `.joblib` model) |
| `app/` | Streamlit prototype: `app.py` + one `pipeline_<name>.py` per member + `common.py` (shared helpers) |
| `report/` | `figures/<name>/`, `failed_cases/<name>/`, `reflections/`, `05_report_figures.ipynb`, `group_results.csv`, `group_method_tests.csv`, `end_to_end_*.csv`, checklist, formulas, workload matrix |
| project folder | `run_all.py`, `evaluate_end_to_end.py`, `run_all_notebooks.bat`, `run_app.bat`, `requirements.txt`, `SETUP.md` |

## How to run

1. Install Python 3.10+, then in this folder: `pip install -r requirements.txt`
2. Open the folder in VS Code (Python + Jupyter extensions), kernel = your Python.
3. For each member, open and **Run All** in this order: `phase1/02_enhancement_<name>` -> `phase2/03_segmentation_<name>` -> `phase3/04_classification_<name>`.
   (`01_slicing_<name>` only if the slices need to be made again; it needs the `.nii.gz` files.)
4. Run `report/05_report_figures.ipynb` for the block diagrams, the group results table and the Wilcoxon comparison.
5. End-to-end check (raw CT -> final tumour mask, incl. the tumour-free slices in `data/slices/<name>_healthy`): `python evaluate_end_to_end.py`
6. Prototype: `streamlit run app/app.py` (Windows: double-click `run_app.bat`). Pick a member, then one slice or a group of slices, from the saved slices or uploaded `.npy` (HU) / `.png` (window 40/400) files. RED = predicted outline, GREEN = ground truth.

Steps 3-5 in one go, on any OS: `python run_all.py` (add `--slicing` to re-make the slices first; Windows: double-click `run_all_notebooks.bat`).
