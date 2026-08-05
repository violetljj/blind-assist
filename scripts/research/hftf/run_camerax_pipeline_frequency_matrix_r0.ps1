param(
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [int]$DurationSeconds = 120,
    [int]$StressSeconds = 5,
    [int]$TtlMs = 750,
    [string]$OutputRoot,
    [switch]$PhaseLockedCadence,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
if ($DurationSeconds -lt 60) { throw "DurationSeconds must be at least 60" }
if ($StressSeconds -lt 3 -or $StressSeconds -ge $DurationSeconds) { throw "StressSeconds is invalid" }

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$runner = Join-Path $PSScriptRoot "run_camerax_latest_only_r0.ps1"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-camerax-pipeline-frequency-matrix-r0-$timestamp"
}
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists: $artifactRoot" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null

$frequencies = @(
    [ordered]@{ hz = 2; period_ms = 500 },
    [ordered]@{ hz = 3; period_ms = 333 },
    [ordered]@{ hz = 4; period_ms = 250 },
    [ordered]@{ hz = 5; period_ms = 200 }
)
$rows = @()
$buildPending = -not $SkipBuild
foreach ($frequency in $frequencies) {
    $runRoot = Join-Path $artifactRoot ("{0}hz" -f $frequency.hz)
    $parameters = @{
        DeviceSerial = $DeviceSerial
        DurationSeconds = $DurationSeconds
        StressSeconds = $StressSeconds
        DepthPeriodMs = $frequency.period_ms
        TtlMs = $TtlMs
        IncludeGeometry = $true
        PipelineGeometry = $true
        OutputRoot = $runRoot
    }
    if ($PhaseLockedCadence) { $parameters.PhaseLockedCadence = $true }
    if (-not $buildPending) { $parameters.SkipBuild = $true }
    & $runner @parameters
    if ($LASTEXITCODE -ne 0) { throw "$($frequency.hz) Hz run failed with exit code $LASTEXITCODE" }
    $buildPending = $false

    $resultPath = Join-Path $runRoot "result.json"
    $bundle = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    $report = $bundle.report
    $failures = @()
    $minimumPacedHz = $frequency.hz * 0.90
    if (-not $report.gate_pass) { $failures += "base gate failed" }
    if (-not $report.pipeline_geometry) { $failures += "pipeline mode was not active" }
    if ($report.max_concurrent_depth_tasks -ne 1) { $failures += "QNN concurrency was not one" }
    if ($report.max_concurrent_geometry_tasks -ne 1) { $failures += "geometry concurrency was not one" }
    if ($report.max_concurrent_pipeline_stages -ne 2) { $failures += "stage overlap was not observed" }
    if ($report.pool_available_after_close -ne 3) { $failures += "YUV pool leak" }
    if ($report.geometry_pool_available_after_close -ne 3) { $failures += "aligned-depth pool leak" }
    if ($report.paced_processed_per_second -lt $minimumPacedHz) {
        $failures += "paced throughput below 90% of requested frequency"
    }
    if ($report.full_depth_geometry_ms.p95 -gt 350.0) { $failures += "full latency P95 exceeded 350 ms" }
    if ($report.fresh_result_age_ms.p95 -gt $TtlMs) { $failures += "fresh age P95 exceeded TTL" }
    if ($report.thermal_fail_closed -ne 0) { $failures += "severe thermal fail-closed occurred" }
    if ($report.failures.Count -ne 0) { $failures += "runtime failures were reported" }

    $rows += [ordered]@{
        requested_hz = $frequency.hz
        period_ms = $frequency.period_ms
        paced_processed_hz = $report.paced_processed_per_second
        paced_processed = $report.paced_processed
        paced_inference_processed = $report.paced_inference_processed
        pending_replaced = $report.pending_replaced
        geometry_pending_replaced = $report.geometry_pending_replaced
        qnn_preprocess_p95_ms = $report.yuv_to_fp16_plus_qnn_ms.p95
        decode_align_p95_ms = $report.fp16_decode_align_ms.p95
        geometry_p95_ms = $report.ground_geometry_ms.p95
        full_pipeline_p95_ms = $report.full_depth_geometry_ms.p95
        fresh_age_p95_ms = $report.fresh_result_age_ms.p95
        geometry_valid = $report.geometry_valid
        geometry_unknown = $report.geometry_unknown
        maximum_thermal_status = $report.maximum_thermal_status
        gate_pass = ($failures.Count -eq 0)
        gate_failures = $failures
        result_path = [IO.Path]::GetRelativePath($repoRoot, $resultPath).Replace('\', '/')
        result_sha256 = (Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash
    }
}

$matrixFailures = @($rows | Where-Object { -not $_.gate_pass } | ForEach-Object {
    "$($_.requested_hz)Hz: $($_.gate_failures -join ', ')"
})
$matrixPath = Join-Path $artifactRoot "matrix.json"
[ordered]@{
    schema = if ($PhaseLockedCadence) {
        "blindassist_dav2_camerax_pipeline_frequency_matrix_r1"
    } else {
        "blindassist_dav2_camerax_pipeline_frequency_matrix_r0"
    }
    generated_at = (Get-Date).ToString("o")
    git_head = (& git -C $repoRoot rev-parse HEAD).Trim()
    device_serial = $DeviceSerial
    duration_seconds_per_frequency = $DurationSeconds
    stress_seconds_per_frequency = $StressSeconds
    ttl_ms = $TtlMs
    phase_locked_cadence = $PhaseLockedCadence.IsPresent
    rows = $rows
    gate_pass = ($matrixFailures.Count -eq 0)
    gate_failures = $matrixFailures
} | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $matrixPath -Encoding utf8

"artifact_root=$artifactRoot"
"matrix_sha256=$((Get-FileHash -LiteralPath $matrixPath -Algorithm SHA256).Hash)"
"gate_pass=$($matrixFailures.Count -eq 0)"
if ($matrixFailures.Count -ne 0) { throw ($matrixFailures -join "; ") }
