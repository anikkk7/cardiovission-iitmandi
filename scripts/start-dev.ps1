$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Local Python environment not found. Running setup first..."
    & (Join-Path $PSScriptRoot "setup.ps1")
}

$backend = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root'; & '$python' -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000"
)

$frontendDir = Join-Path $root "frontend"
$frontend = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$frontendDir'; npm run dev -- --host 127.0.0.1"
)

$pidFile = Join-Path $root ".dev-pids.json"
@{
    backend = $backend.Id
    frontend = $frontend.Id
} | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8

Write-Host "CardioVision is starting..."
Write-Host "Backend:  http://127.0.0.1:8000/docs"
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host "Backend process: $($backend.Id)"
Write-Host "Frontend process: $($frontend.Id)"
Write-Host "Run scripts\stop-dev.ps1 to stop both processes."
