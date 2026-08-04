param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$QairtRoot = "E:\codex-tools\qairt\2.47.0.260601",
    [string]$CachedDlcPath = "/data/local/tmp/ba_qairt_htp_r0/dav2-metric/518x686/output/model-sm8650-cached.dlc",
    [string]$OutputRoot,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_camerax_latest_only_r0.ps1"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\camerax-sustained-10min-r0-$timestamp"
}
$previousStayOn = (& $AdbPath -s $DeviceSerial shell settings get global stay_on_while_plugged_in 2>$null).Trim()
try {
    & $AdbPath -s $DeviceSerial shell settings put global stay_on_while_plugged_in 2
    & $AdbPath -s $DeviceSerial shell input keyevent KEYCODE_WAKEUP
    $parameters = @{
        AdbPath = $AdbPath; DeviceSerial = $DeviceSerial; QairtRoot = $QairtRoot
        CachedDlcPath = $CachedDlcPath; DurationSeconds = 600; StressSeconds = 5
        DepthPeriodMs = 500; TtlMs = 750; OutputRoot = $artifactRoot; IncludeGeometry = $true
    }
    if ($SkipBuild) { $parameters.SkipBuild = $true }
    & $runner @parameters
    if ($LASTEXITCODE -ne 0) { throw "ten-minute CameraX runner failed with exit code $LASTEXITCODE" }
} finally {
    if ($previousStayOn -match '^\d+$') {
        & $AdbPath -s $DeviceSerial shell settings put global stay_on_while_plugged_in $previousStayOn
    }
}
