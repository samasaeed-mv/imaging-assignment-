@echo off
cd /d "%~dp0"
echo Starting the prototype app... (close this window to stop it)
python -m streamlit run app/app.py
pause
