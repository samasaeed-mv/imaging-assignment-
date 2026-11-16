# Automated Abdominal CT Organ & Lesion Diagnostics Pipeline (LiTS, liver + liver tumour)

Classical image processing only (no deep learning / TensorFlow / pre-trained models).
Raw CT slice -> Phase 1 enhancement -> Phase 2 segmentation -> Phase 3 features + classification -> diagnosis (healthy vs tumour).

## Who did what

| Member | Organ | Patient | Phase 1 | Phase 2 (A vs B) | Phase 3 classifier |
|---|---|---|---|---|---|
| charvi | liver | 1 | anisotropic diffusion vs Gaussian, CLAHE vs HE, homomorphic | adaptive threshold vs **seeded region growing** | k-NN |
| sama | liver | 0 | bilateral vs median, CLAHE vs HE, Butterworth | Multi-Otsu vs **marker-controlled watershed** | SVM |
| kritikaa | tumour | 22 | bilateral vs Gaussian, CLAHE vs HE, Butterworth | Multi-Otsu vs **seeded region growing** (inside liver) | random forest |
| evangeline | tumour | 29 | anisotropic diffusion vs median, CLAHE vs HE, homomorphic | adaptive threshold vs **marker-controlled watershed** (inside liver) | decision tree |

Bold = the better method, used in the app. Every file, folder, figure and failed case carries the member's name.

## Folders

| Folder | What is in it |
|---|---|
| `data/raw_volumes/` | LiTS `volume-N.nii.gz` + `segmentation-N.nii.gz` (gzipped originals) |
| `data/slices/<name>/` | each member's 2D test slices: `*_ct.npy` (raw HU), `*_liver.png`, `*_tumour.png` |
| `data/data_log.csv` | which patient and slices each member used |
| `phase1/` | `01_slicing_<name>`, `02_enhancement_<name>` notebooks + `outputs_<name>/` (enhanced slices, PSNR/CNR table) |
| `phase2/` | `03_segmentation_<name>` notebooks + `outputs_<name>/` (predicted masks, Dice/IoU table) |
| `phase3/` | `04_classification_<name>` notebooks + `outputs_<name>/` (features, metrics, trained model) |
| `app/` | Streamlit prototype: `app.py` + one `pipeline_<name>.py` per member + `common.py` (shared helpers) |
| `report/` | `figures/<name>/`, `failed_cases/<name>/`, `05_report_figures.ipynb`, `group_results.csv`, checklist, formulas, workload matrix, reflections |

## How to run

1. Install Python 3.10+, then in this folder: `pip install -r requirements.txt`
2. Open the folder in VS Code (Python + Jupyter extensions), kernel = your Python.
3. For each member, open and **Run All** in this order: `phase1/02_enhancement_<name>` -> `phase2/03_segmentation_<name>` -> `phase3/04_classification_<name>`.
   (`01_slicing_<name>` only if the slices need to be made again; it needs the `.nii.gz` files.)
4. Run `report/05_report_figures.ipynb` for the block diagrams, the group results table and the Wilcoxon comparison.
5. End-to-end check (raw CT -> final tumour mask, incl. the tumour-free slices in `data/slices/<name>_healthy`): `python evaluate_end_to_end.py`
6. Prototype: `streamlit run app/app.py`, pick a member and one slice or a group of slices. RED = predicted outline, GREEN = ground truth.

Steps 3-5 in one go, on any OS: `python run_all.py` (add `--slicing` to re-make the slices first; Windows: double-click `run_all_notebooks.bat`).
