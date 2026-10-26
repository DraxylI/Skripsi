# Skrip PowerShell untuk inisialisasi Virtual Environment Pipeline OULAD
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Menyiapkan Virtual Environment untuk Pipeline OULAD..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Membuat virtual environment .venv..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "Virtual environment .venv sudah ada." -ForegroundColor Green
}

Write-Host "Mengaktifkan .venv dan menginstal dependencies..." -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "Virtual environment siap digunakan!" -ForegroundColor Green
Write-Host "Untuk menjalankan pipeline:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python run_pipeline.py all" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Green

