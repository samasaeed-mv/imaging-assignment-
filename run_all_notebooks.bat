@echo off
rem Windows double-click shortcut. The real work is in run_all.py (works on any OS: python run_all.py)
cd /d "%~dp0"
echo Running all notebooks + the end-to-end evaluation (this takes a while)...
python run_all.py
echo.
echo Finished. See run_all_log.txt for details.
pause
