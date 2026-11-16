# IPCV Assignment - Liver CT (LiTS) - Setup and Step 1

Due: 13 Dec 2026 | Dataset: LiTS (Option A in the brief) | Rules: classical methods only, no deep learning / TensorFlow / pre-trained models.

## Folders
- data/raw_volumes/  the downloaded LiTS files (volume-N.nii + segmentation-N.nii)
- data/slices/       your 2D slices + ground-truth masks (one subfolder per member)
- data/data_log.csv  which patient + slices each member used (needed for the report)
- phase1/            enhanced images, spectrum images, PSNR/CNR results
- phase2/            your segmentation masks, overlays, Dice/IoU results
- phase3/            feature tables, classifier results, confusion matrices
- app/               the prototype (red = your mask, green = ground truth)
- report/figures/    images you will put in the report
- report/failed_cases/  images where your method failed + a note on why (critical analysis = 20%)

## 1. Install Python (Windows, once)
1. Install Python 3.10 or newer from python.org. On the first installer screen, tick "Add python.exe to PATH".
2. Open Command Prompt in THIS folder (in File Explorer, click the address bar, type cmd, press Enter).
3. Run:  pip install -r requirements.txt
4. Start Jupyter:  jupyter notebook   (or open the folder in VS Code and install the "Python" and "Jupyter" extensions)
5. Check it works: in a new notebook cell run  import cv2, skimage, nibabel, sklearn  - no error = ready.
If your laptop is too slow, everything also works in Google Colab (upload this folder to Google Drive).

## 2. Shared group folder
Make ONE shared Google Drive folder (or GitHub repo) with this same layout, so everyone's code and results end up in the same structure for the prototype and report.

## 3. Step 1 - Get the data
1. Open the dataset link from the brief: https://github.com/Auggen21/LITS-Challenge
2. Use its SMALLER SUBSET on Google Drive (linked in that README). Do NOT download the full ~50 GB set.
3. Put the files in data/raw_volumes/. Each patient = two files: volume-N.nii (the CT) and segmentation-N.nii (the answer mask: 0 = background, 1 = liver, 2 = tumour).
4. As a group, give each member DIFFERENT patient numbers (the brief requires different cases per member). Write them in data/data_log.csv.

## 4. Step 1 - Turn 3D scans into 2D slices (write this code yourself)
Goal: for each of your patients, save a few axial slices as images, each with its liver mask and tumour mask.
- Load a volume and its segmentation with nibabel.load(...).get_fdata()  -> 3D arrays (rows, cols, slices).
- Loop over the slice axis. Keep slices where the segmentation contains tumour (value 2) - and liver (value 1) for liver members.
- For each kept slice save: the CT slice (keep the raw HU values as .npy; windowing to 0-255 is Phase 1), the liver mask (seg >= 1) and the tumour mask (seg == 2) as PNG.
- The slices may appear sideways: np.rot90 fixes the orientation. Use the same rotation for image and masks.
- Aim for at least 10 test images per organ, plus extra slices for training the classifier in Phase 3.
- Log every patient + slice number in data/data_log.csv.
