@echo off
echo ========================================================
echo Menyiapkan Virtual Environment untuk Pipeline OULAD...
echo ========================================================

cd /d "%~dp0"

if not exist ".venv" (
    echo Membuat virtual environment .venv...
    python -m venv .venv
) else (
    echo Virtual environment .venv sudah ada.
)

echo Mengaktifkan .venv dan menginstal dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================================
echo Virtual environment siap digunakan!
echo Untuk menjalankan pipeline, gunakan:
echo   .venv\Scripts\activate.bat
echo   python run_pipeline.py all
echo ========================================================
pause

