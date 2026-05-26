param(
    [int]$ImageLimit = 100,
    [int]$PureWarmup = 10,
    [int]$PureRuns = 100,
    [int]$AppRunsPerImage = 1,
    [int]$DefaultRegressionSeconds = 90,
    [switch]$SkipDefaultRegression,
    [string]$AdbPath
)

$ErrorActionPreference = "Stop"

$script = Join-Path $PSScriptRoot "run_detector_ab_device_benchmark.ps1"
$arguments = @(
    "-ImageLimit", "$ImageLimit",
    "-PureWarmup", "$PureWarmup",
    "-PureRuns", "$PureRuns",
    "-AppRunsPerImage", "$AppRunsPerImage",
    "-DefaultRegressionSeconds", "$DefaultRegressionSeconds"
)
if ($SkipDefaultRegression) {
    $arguments += "-SkipDefaultRegression"
}
if ($AdbPath) {
    $arguments += @("-AdbPath", $AdbPath)
}

Write-Host "run_yolo26n_device_benchmark.ps1 is now a compatibility wrapper for the detector A/B benchmark."
& powershell -ExecutionPolicy Bypass -File $script @arguments
exit $LASTEXITCODE
