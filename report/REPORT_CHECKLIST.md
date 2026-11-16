# Report checklist (brief vs what we have)

Report: ~5000 words, APA 7. Due **13 Dec 2026**. Marks: intro/literature/justification 20%, implementation + code + methodology 30%, results + discussion 20%, critical analysis 20%, overall report 10%; prototype (video, demo, presentation, individual contribution) 20%.

## Report sections

| Section | Status | Material we already have |
|---|---|---|
| Cover page, table of contents | to write | names + TP numbers |
| Signed contribution / workload matrix | template ready | `report/workload_matrix.md` |
| Abstract (200-300 words) | to write | `report/group_results.csv` for the key numbers |
| Introduction | to write | problem: liver tumours in CT, why automate, LiTS dataset |
| Short literature review | to write | classical methods: windowing, bilateral/anisotropic, CLAHE, Otsu, watershed, region growing, GLCM, SVM/k-NN/RF |
| Method justification | to write | why each member's methods; `report/figures/<name>/<name>_pipeline_diagram.png`, `report/figures/group_pipeline_diagram.png` |
| Formulas | ready to copy | `report/formulas.md` |
| Results | material ready | per member: PSNR/CNR table (phase1), Dice/IoU table (phase2), confusion matrix + metrics (phase3), `group_results.csv` |
| Raw vs enhanced images | ready | `<name>_p1_windowing / _spatial_filters / _contrast / _frequency.png` |
| Overlay masks (red vs green) | ready | `<name>_p2_overlays.png`, app screenshots |
| GLCM tables (0/45/90/135) | ready | `report/figures/<name>/<name>_glcm_table.csv` |
| Discussion | to write | compare members, liver vs tumour difficulty |
| Critical analysis + FAILED cases | material ready | `report/failed_cases/<name>/<name>_phase2_failed_cases.md`, `<name>_phase3_failed_cases.md` + overlay images |
| Future work | to write | more patients, patient-wise split (no neighbouring-slice leakage), 3D context, tuning per patient |
| Conclusion | to write | |
| References (APA, scholarly) | to write | |
| AI declaration form | to fill | list which code/text had AI help (keep within the allowed share) |

## Per member

| | charvi | sama | kritikaa | evangeline |
|---|---|---|---|---|
| 10+ test images | 18 slices | 16 slices | 12 slices | 22 slices |
| Phase 1 notebook + figures + PSNR/CNR | yes | yes | yes | yes |
| Phase 2 notebook + 2 methods + morphology + Dice/IoU | yes | yes | yes | yes |
| Phase 3 notebook + shape/GLCM/first-order + classifier + 70/30 + 5-fold + metrics | yes | yes | yes | yes |
| Failed cases (phase 2 + phase 3) | yes | yes | yes | yes |
| Pipeline block diagram | yes | yes | yes | yes |
| App pipeline (`app/pipeline_<name>.py`) | yes | yes | yes | yes |
| 1-page reflection | template | template | template | template |
| 10-15 min video (slides, live demo, code walk-through) | to record | to record | to record | to record |

## To check with the lecturer
- Some algorithm *types* are shared between two members (e.g. watershed: sama + evangeline) because the brief offers only two options per step; every member has a different combination and a different organ/patient. Confirm this is acceptable.
- Classifier test split is at region level, so neighbouring slices of the same tumour can appear in train and test (optimistic scores). Mention in critical analysis.
