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
    Join-Path $repoRoot "artifacts.local\evidence\hftf\camerax-canonical-sustained-10min-r1-$timestamp"
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
    if ($LASTEXITCODE -ne 0) { throw "canonical ten-minute CameraX runner failed with exit code $LASTEXITCODE" }

    $resultPath = Join-Path $artifactRoot "result.json"
    $bundle = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    $actualRoute = $bundle.report.contract.preprocess_route
    $expectedRoute = "canonical_native_official_fp32_then_integer_rnte_fp16_v1"
    $gateFailures = @()
    if ($actualRoute -ne $expectedRoute) { $gateFailures += "unexpected preprocess route: $actualRoute" }
    if (-not $bundle.report.gate_pass) { $gateFailures += "base CameraX sustained gate failed" }
    if ($bundle.report.yuv_to_fp16_plus_qnn_ms.p95 -gt 250.0) { $gateFailures += "preprocess plus QNN P95 exceeded 250 ms" }
    if ($bundle.report.full_depth_geometry_ms.p95 -gt 350.0) { $gateFailures += "full pipeline P95 exceeded 350 ms" }
    if ($bundle.report.fresh_result_age_ms.p95 -gt 750.0) { $gateFailures += "fresh result age P95 exceeded 750 ms" }
    [ordered]@{
        schema = "blindassist_camerax_canonical_sustained_gate_r1"
        expected_preprocess_route = $expectedRoute
        observed_preprocess_route = $actualRoute
        thresholds = [ordered]@{
            preprocess_plus_qnn_p95_ms_max = 250.0
            full_pipeline_p95_ms_max = 350.0
            fresh_result_age_p95_ms_max = 750.0
        }
        observed = [ordered]@{
            preprocess_plus_qnn_p95_ms = $bundle.report.yuv_to_fp16_plus_qnn_ms.p95
            full_pipeline_p95_ms = $bundle.report.full_depth_geometry_ms.p95
            fresh_result_age_p95_ms = $bundle.report.fresh_result_age_ms.p95
        }
        gate_pass = ($gateFailures.Count -eq 0)
        gate_failures = $gateFailures
        result_sha256 = (Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (
        Join-Path $artifactRoot "canonical-sustained-gate.json"
    ) -Encoding utf8
    if ($gateFailures.Count -ne 0) { throw ($gateFailures -join "; ") }
} finally {
    if ($previousStayOn -match '^\d+$') {
        & $AdbPath -s $DeviceSerial shell settings put global stay_on_while_plugged_in $previousStayOn
    }
}
