@echo off
echo ===== ScamSense setup (needs internet ONCE, to install libraries) =====
python --version || (echo Python is not installed. Get it free from python.org and tick "Add Python to PATH". & pause & exit /b)
pip install -r requirements.txt
python run_all.py
echo.
echo ===== Starting the app (works offline from now on) =====
python app.py
pause
