@echo off
cd /d "%~dp0"
echo Running all notebooks (this takes a few minutes)... > run_all_log.txt
echo ==== phase1\02_enhancement_charvi.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase1\02_enhancement_charvi.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase1\02_enhancement_charvi.ipynb >> run_all_log.txt) else (echo OK: phase1\02_enhancement_charvi.ipynb >> run_all_log.txt)
echo ==== phase2\03_segmentation_charvi.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase2\03_segmentation_charvi.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase2\03_segmentation_charvi.ipynb >> run_all_log.txt) else (echo OK: phase2\03_segmentation_charvi.ipynb >> run_all_log.txt)
echo ==== phase3\04_classification_charvi.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase3\04_classification_charvi.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase3\04_classification_charvi.ipynb >> run_all_log.txt) else (echo OK: phase3\04_classification_charvi.ipynb >> run_all_log.txt)
echo ==== phase1\02_enhancement_sama.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase1\02_enhancement_sama.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase1\02_enhancement_sama.ipynb >> run_all_log.txt) else (echo OK: phase1\02_enhancement_sama.ipynb >> run_all_log.txt)
echo ==== phase2\03_segmentation_sama.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase2\03_segmentation_sama.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase2\03_segmentation_sama.ipynb >> run_all_log.txt) else (echo OK: phase2\03_segmentation_sama.ipynb >> run_all_log.txt)
echo ==== phase3\04_classification_sama.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase3\04_classification_sama.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase3\04_classification_sama.ipynb >> run_all_log.txt) else (echo OK: phase3\04_classification_sama.ipynb >> run_all_log.txt)
echo ==== phase1\02_enhancement_kritikaa.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase1\02_enhancement_kritikaa.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase1\02_enhancement_kritikaa.ipynb >> run_all_log.txt) else (echo OK: phase1\02_enhancement_kritikaa.ipynb >> run_all_log.txt)
echo ==== phase2\03_segmentation_kritikaa.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase2\03_segmentation_kritikaa.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase2\03_segmentation_kritikaa.ipynb >> run_all_log.txt) else (echo OK: phase2\03_segmentation_kritikaa.ipynb >> run_all_log.txt)
echo ==== phase3\04_classification_kritikaa.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase3\04_classification_kritikaa.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase3\04_classification_kritikaa.ipynb >> run_all_log.txt) else (echo OK: phase3\04_classification_kritikaa.ipynb >> run_all_log.txt)
echo ==== phase1\02_enhancement_evangeline.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase1\02_enhancement_evangeline.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase1\02_enhancement_evangeline.ipynb >> run_all_log.txt) else (echo OK: phase1\02_enhancement_evangeline.ipynb >> run_all_log.txt)
echo ==== phase2\03_segmentation_evangeline.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase2\03_segmentation_evangeline.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase2\03_segmentation_evangeline.ipynb >> run_all_log.txt) else (echo OK: phase2\03_segmentation_evangeline.ipynb >> run_all_log.txt)
echo ==== phase3\04_classification_evangeline.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "phase3\04_classification_evangeline.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: phase3\04_classification_evangeline.ipynb >> run_all_log.txt) else (echo OK: phase3\04_classification_evangeline.ipynb >> run_all_log.txt)
echo ==== report\05_report_figures.ipynb >> run_all_log.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "report\05_report_figures.ipynb" >> run_all_log.txt 2>&1
if errorlevel 1 (echo FAILED: report\05_report_figures.ipynb >> run_all_log.txt) else (echo OK: report\05_report_figures.ipynb >> run_all_log.txt)
echo ALL DONE >> run_all_log.txt
echo.
echo All notebooks finished. See run_all_log.txt
pause
