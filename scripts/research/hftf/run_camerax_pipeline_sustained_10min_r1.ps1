param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$OutputRoot,
    [switch]$NativeFp16Decode,
    [string]$Fp16ParityResultPath,
    [switch]$NativeGeometry,
    [string]$NativeGeometryParityResultPath,
    [switch]$NativeDirectDepthBridge,
    [string]$DirectDepthBridgeParityResultPath,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$runner = Join-Path $PSScriptRoot "run_camerax_latest_only_r0.ps1"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-camerax-pipeline-sustained-10min-r1-$timestamp"
}
$previousStayOn = (& $AdbPath -s $DeviceSerial shell settings get global stay_on_while_plugged_in 2>$null).Trim()
try {
    & $AdbPath -s $DeviceSerial shell settings put global stay_on_while_plugged_in 2
    & $AdbPath -s $DeviceSerial shell input keyevent KEYCODE_WAKEUP
    $parameters = @{
        AdbPath = $AdbPath
        DeviceSerial = $DeviceSerial
        DurationSeconds = 600
        StressSeconds = 5
        DepthPeriodMs = 200
        TtlMs = 750
        IncludeGeometry = $true
        PipelineGeometry = $true
        PhaseLockedCadence = $true
        OutputRoot = $artifactRoot
    }
    if ($NativeFp16Decode) { $parameters.NativeFp16Decode = $true }
    if ($NativeGeometry) { $parameters.NativeGeometry = $true }
    if ($NativeDirectDepthBridge) { $parameters.NativeDirectDepthBridge = $true }
    if ($SkipBuild) { $parameters.SkipBuild = $true }
    & $runner @parameters
    if ($LASTEXITCODE -ne 0) { throw "pipeline ten-minute runner failed with exit code $LASTEXITCODE" }

    $resultPath = Join-Path $artifactRoot "result.json"
    $bundle = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    $report = $bundle.report
    $failures = @()
    if (-not $report.gate_pass) { $failures += "base CameraX gate failed" }
    if (-not $report.pipeline_geometry) { $failures += "pipeline mode was not active" }
    if (-not $report.phase_locked_cadence) { $failures += "phase-locked cadence was not active" }
    if ($NativeFp16Decode -and -not $report.native_fp16_decode) { $failures += "native FP16 decode was not active" }
    if ($NativeGeometry -and -not $report.native_geometry) { $failures += "native geometry was not active" }
    if ($NativeDirectDepthBridge -and -not $report.native_direct_depth_bridge) { $failures += "native direct depth bridge was not active" }
    if ($report.paced_processed_per_second -lt 4.5) { $failures += "5 Hz paced throughput was below 4.5 Hz" }
    if ($report.max_concurrent_depth_tasks -ne 1) { $failures += "QNN concurrency was not one" }
    if ($report.max_concurrent_geometry_tasks -ne 1) { $failures += "geometry concurrency was not one" }
    if ($report.max_concurrent_pipeline_stages -ne 2) { $failures += "stage overlap was not observed" }
    if ($report.pool_available_after_close -ne 3) { $failures += "YUV pool leak" }
    if ($report.geometry_pool_available_after_close -ne 3) { $failures += "aligned-depth pool leak" }
    if ($report.full_depth_geometry_ms.p95 -gt 350.0) { $failures += "full latency P95 exceeded 350 ms" }
    if ($report.fresh_result_age_ms.p95 -gt 750.0) { $failures += "fresh age P95 exceeded TTL" }
    if ($report.thermal_fail_closed -ne 0) { $failures += "severe thermal fail-closed occurred" }
    if ($report.noninteractive_camera_observations -ne 0) { $failures += "screen was not continuously interactive" }
    if ($report.failures.Count -ne 0) { $failures += "runtime failures were reported" }

    $gatePath = Join-Path $artifactRoot "pipeline-sustained-gate.json"
    $parityBinding = $null
    if ($NativeFp16Decode) {
        if (-not $Fp16ParityResultPath) { throw "Fp16ParityResultPath is required for native decode" }
        $parityCandidate = if ([IO.Path]::IsPathRooted($Fp16ParityResultPath)) {
            $Fp16ParityResultPath
        } else {
            Join-Path $repoRoot $Fp16ParityResultPath
        }
        $resolvedParity = (Resolve-Path -LiteralPath $parityCandidate).Path
        $parity = Get-Content -LiteralPath $resolvedParity -Raw | ConvertFrom-Json
        if (-not $parity.report.pass -or $parity.report.bit_patterns -ne 65536 -or
            $parity.report.mismatches -ne 0) {
            throw "native FP16 decode parity receipt is not a full-domain pass"
        }
        $parityBinding = [ordered]@{
            path = [IO.Path]::GetRelativePath($repoRoot, $resolvedParity).Replace('\', '/')
            sha256 = (Get-FileHash -LiteralPath $resolvedParity -Algorithm SHA256).Hash
            bit_patterns = $parity.report.bit_patterns
            mismatches = $parity.report.mismatches
        }
    }
    $geometryParityBinding = $null
    if ($NativeGeometry) {
        if (-not $NativeGeometryParityResultPath) { throw "NativeGeometryParityResultPath is required for native geometry" }
        $geometryParityCandidate = if ([IO.Path]::IsPathRooted($NativeGeometryParityResultPath)) {
            $NativeGeometryParityResultPath
        } else {
            Join-Path $repoRoot $NativeGeometryParityResultPath
        }
        $resolvedGeometryParity = (Resolve-Path -LiteralPath $geometryParityCandidate).Path
        $geometryParity = Get-Content -LiteralPath $resolvedGeometryParity -Raw | ConvertFrom-Json
        if (-not $geometryParity.report.gate_pass) {
            throw "native geometry parity receipt did not pass"
        }
        $geometryParityBinding = [ordered]@{
            path = [IO.Path]::GetRelativePath($repoRoot, $resolvedGeometryParity).Replace('\', '/')
            sha256 = (Get-FileHash -LiteralPath $resolvedGeometryParity -Algorithm SHA256).Hash
            cases = $geometryParity.report.cases.Count
            maximum_absolute_field_error = ($geometryParity.report.cases | Measure-Object -Property maximum_absolute_field_error -Maximum).Maximum
        }
    }
    $directDepthParityBinding = $null
    if ($NativeDirectDepthBridge) {
        if (-not $DirectDepthBridgeParityResultPath) { throw "DirectDepthBridgeParityResultPath is required for native direct depth bridge" }
        $directDepthParityCandidate = if ([IO.Path]::IsPathRooted($DirectDepthBridgeParityResultPath)) {
            $DirectDepthBridgeParityResultPath
        } else {
            Join-Path $repoRoot $DirectDepthBridgeParityResultPath
        }
        $resolvedDirectDepthParity = (Resolve-Path -LiteralPath $directDepthParityCandidate).Path
        $directDepthParity = Get-Content -LiteralPath $resolvedDirectDepthParity -Raw | ConvertFrom-Json
        if (-not $directDepthParity.report.gate_pass -or
            ($directDepthParity.report.cases | Measure-Object -Property finite_raw_bit_mismatches -Sum).Sum -ne 0 -or
            ($directDepthParity.report.cases | Measure-Object -Property nonfinite_class_mismatches -Sum).Sum -ne 0) {
            throw "native direct depth bridge parity receipt did not pass exact aligned-depth parity"
        }
        $directDepthParityBinding = [ordered]@{
            path = [IO.Path]::GetRelativePath($repoRoot, $resolvedDirectDepthParity).Replace('\', '/')
            sha256 = (Get-FileHash -LiteralPath $resolvedDirectDepthParity -Algorithm SHA256).Hash
            cases = $directDepthParity.report.cases.Count
            finite_raw_bit_mismatches = ($directDepthParity.report.cases | Measure-Object -Property finite_raw_bit_mismatches -Sum).Sum
            nonfinite_class_mismatches = ($directDepthParity.report.cases | Measure-Object -Property nonfinite_class_mismatches -Sum).Sum
        }
    }
    $boundSources = @(
        "core/vision/src/main/java/com/linnan/blindassist/vision/LatestOnlySidecar.kt",
        "core/vision/src/main/java/com/linnan/blindassist/vision/PhaseLockedCadenceGate.kt",
        "hftf-device-canary/src/main/java/com/linnan/blindassist/hftf/CameraXLatestOnlyDepthDeviceTest.kt",
        "hftf-device-canary/src/main/java/com/linnan/blindassist/hftf/Dav2QnnCachedContext.kt",
        "hftf-device-canary/src/main/java/com/linnan/blindassist/hftf/Dav2Preprocessors.kt",
        "hftf-device-canary/src/main/java/com/linnan/blindassist/hftf/Dav2NativeGeometry.kt",
        "hftf-device-canary/src/main/java/com/linnan/blindassist/hftf/Dav2Yuv420RgbConverter.kt",
        "hftf-device-canary/src/main/cpp/dav2_preprocess_native.cpp",
        "hftf-device-canary/src/main/cpp/dav2_native_geometry.cpp",
        "hftf-metric-depth-canary-core/src/main/kotlin/com/linnan/blindassist/hftf/metricdepth/KnownHeightGroundPipeline.kt",
        "scripts/research/hftf/run_camerax_latest_only_r0.ps1",
        "scripts/research/hftf/run_camerax_pipeline_sustained_10min_r1.ps1"
    )
    $sourceHashes = [ordered]@{}
    foreach ($relativePath in $boundSources) {
        $sourceHashes[$relativePath] = (Get-FileHash -LiteralPath (
            Join-Path $repoRoot $relativePath
        ) -Algorithm SHA256).Hash
    }
    [ordered]@{
        schema = if ($NativeDirectDepthBridge) {
            "blindassist_dav2_camerax_pipeline_native_direct_depth_sustained_gate_r4"
        } elseif ($NativeGeometry) {
            "blindassist_dav2_camerax_pipeline_native_geometry_sustained_gate_r3"
        } elseif ($NativeFp16Decode) {
            "blindassist_dav2_camerax_pipeline_native_decode_sustained_gate_r2"
        } else {
            "blindassist_dav2_camerax_pipeline_sustained_gate_r1"
        }
        generated_at = (Get-Date).ToString("o")
        git_head = $bundle.git_head
        device_serial = $DeviceSerial
        native_fp16_decode = $NativeFp16Decode.IsPresent
        native_geometry = $NativeGeometry.IsPresent
        native_direct_depth_bridge = $NativeDirectDepthBridge.IsPresent
        fp16_decode_parity = $parityBinding
        native_geometry_parity = $geometryParityBinding
        direct_depth_bridge_parity = $directDepthParityBinding
        source_sha256 = $sourceHashes
        requested_hz = 5
        thresholds = [ordered]@{
            paced_processed_hz_min = 4.5
            full_pipeline_p95_ms_max = 350.0
            fresh_result_age_p95_ms_max = 750.0
        }
        observed = [ordered]@{
            paced_processed_hz = $report.paced_processed_per_second
            full_pipeline_p95_ms = $report.full_depth_geometry_ms.p95
            fresh_result_age_p95_ms = $report.fresh_result_age_ms.p95
            maximum_thermal_status = $report.maximum_thermal_status
            geometry_valid = $report.geometry_valid
            geometry_unknown = $report.geometry_unknown
            geometry_pending_replaced = $report.geometry_pending_replaced
        }
        gate_pass = ($failures.Count -eq 0)
        gate_failures = $failures
        result_sha256 = (Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash
    } | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $gatePath -Encoding utf8
    "artifact_root=$artifactRoot"
    "gate_sha256=$((Get-FileHash -LiteralPath $gatePath -Algorithm SHA256).Hash)"
    "gate_pass=$($failures.Count -eq 0)"
    if ($failures.Count -ne 0) { throw ($failures -join "; ") }
} finally {
    if ($previousStayOn -match '^\d+$') {
        & $AdbPath -s $DeviceSerial shell settings put global stay_on_while_plugged_in $previousStayOn
    }
}
