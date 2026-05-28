@echo off
echo ============================================================
echo   Sign Language Model Training
echo   Dataset: ..\asl_dataset
echo ============================================================
echo.

REM Activate venv if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [INFO] venv activated.
) else (
    echo [WARN] No venv found — using system Python
)

REM Show which python we are using
echo [INFO] Python path:
where python
echo [INFO] Python version:
python --version
echo.

REM Make sure model/ directory exists
if not exist model mkdir model

REM Verify numpy is reachable (quick sanity check)
python -c "import numpy; print('[INFO] numpy', numpy.__version__)"
if errorlevel 1 (
    echo [ERROR] numpy not found. Run:  pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [Training] Starting — this may take 5-20 minutes ...
echo.

python -u train_model.py --data_dir "..\asl_dataset" --epochs 30 --max_per_class 300
set TRAIN_EXIT=%ERRORLEVEL%

echo.
if %TRAIN_EXIT% NEQ 0 (
    echo [ERROR] Training exited with code %TRAIN_EXIT%. See errors above.
    echo TIP: If you see an ImportError for tensorflow or mediapipe, run:
    echo      pip install tensorflow==2.17.0 mediapipe==0.10.14
) else (
    echo ============================================================
    echo   Training complete!  Restart Flask:  python app.py
    echo ============================================================
)
pause
