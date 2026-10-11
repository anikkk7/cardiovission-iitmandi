$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root ".dev-pids.json"

function Stop-ProcessTree {
    param([int]$ProcessId)

    $children = Get-CimInstance Win32_Process |
        Where-Object { $_.ParentProcessId -eq $ProcessId }

    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId $child.ProcessId
    }

    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Stop-Process -Id $ProcessId -Force
        Write-Host "Stopped process $ProcessId"
    }
}

if (Test-Path $pidFile) {
    $pids = Get-Content $pidFile -Raw | ConvertFrom-Json
    foreach ($id in @($pids.backend, $pids.frontend)) {
        if ($id) {
            Stop-ProcessTree -ProcessId $id
        }
    }
    Remove-Item $pidFile -Force
} else {
    Write-Host "No saved dev process file found. Checking standard ports."
}

foreach ($port in @(8000, 5173)) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        Where-Object { $_ } |
        ForEach-Object { Stop-ProcessTree -ProcessId $_ }
}
