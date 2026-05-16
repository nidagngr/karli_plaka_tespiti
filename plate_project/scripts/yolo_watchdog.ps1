$ErrorActionPreference = "Continue"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$evalDir = Join-Path $projectRoot "outputs\evaluation"
New-Item -ItemType Directory -Path $evalDir -Force | Out-Null

$stdoutLog = Join-Path $evalDir "yolo_train_stdout.log"
$stderrLog = Join-Path $evalDir "yolo_train_stderr.log"
$watchdogLog = Join-Path $evalDir "yolo_watchdog.log"
$pidFile = Join-Path $evalDir "yolo_watchdog.pid"
$summaryFile = Join-Path $evalDir "yolo_train_summary.json"
$maxRestarts = 999

function Write-WatchdogLog([string]$msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $watchdogLog -Value "$ts | $msg"
}

Write-WatchdogLog "YOLO watchdog started."

if (Test-Path $pidFile) {
    $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        Write-WatchdogLog "Another watchdog instance is already running (PID=$existingPid). Exiting."
        exit 0
    }
}

Set-Content -Path $pidFile -Value $PID

for ($attempt = 1; $attempt -le $maxRestarts; $attempt++) {
    if (Test-Path $summaryFile) {
        Write-WatchdogLog "Training completed marker found. Stopping watchdog."
        exit 0
    }

    Write-WatchdogLog "Starting training attempt $attempt."
    & python src/train_yolo.py --config configs/yolo_config.yaml 1>> $stdoutLog 2>> $stderrLog
    $exitCode = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { -1 }

    if (Test-Path $summaryFile) {
        Write-WatchdogLog "Training completed successfully on attempt $attempt."
        exit 0
    }

    Write-WatchdogLog "Training stopped with exit code $exitCode on attempt $attempt. Restarting in 10 seconds."
    Start-Sleep -Seconds 10
}

Write-WatchdogLog "Max restart count reached without completion."
if (Test-Path $pidFile) { Remove-Item $pidFile -Force }
exit 1
