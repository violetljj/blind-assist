[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Container })]
    [string]$DatasetRoot,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$Weights,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$Python = 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe',

    [ValidateRange(1, 8580)]
    [int]$MaxFrames = 8,

    [ValidateSet('uniform', 'head')]
    [string]$Selection = 'uniform',

    [string]$SelectionContract,

    [ValidateRange(1, 32)]
    [int]$Batch = 1,

    [ValidateRange(128, 640)]
    [int]$ImageSize = 256,

    [ValidateRange(0.05, 0.75)]
    [double]$MemoryFraction = 0.15,

    [ValidateRange(0, 5000)]
    [int]$InterBatchDelayMs = 0,

    [ValidateSet('full_frame', 'full_plus_4_corner_crops')]
    [string]$InferenceMode = 'full_frame',

    [ValidateRange(0, 1000)]
    [int]$InterViewDelayMs = 0,

    [ValidateRange(50, 85)]
    [int]$MaxTemperatureC = 72,

    [ValidateRange(30, 7200)]
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$repoRoot = Split-Path -Parent $PSScriptRoot
$benchmark = Join-Path $PSScriptRoot 'benchmark_revel_yolo_person_detector.py'
$nvidiaSmi = (Get-Command nvidia-smi.exe -ErrorAction Stop).Source
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
$resolvedDataset = (Resolve-Path -LiteralPath $DatasetRoot).Path
$resolvedWeights = (Resolve-Path -LiteralPath $Weights).Path
$resolvedPython = (Resolve-Path -LiteralPath $Python).Path
$resolvedSelectionContract = if ($SelectionContract) { (Resolve-Path -LiteralPath $SelectionContract -ErrorAction Stop).Path } else { $null }

if (Test-Path -LiteralPath $resolvedOutput) {
    if (Get-ChildItem -LiteralPath $resolvedOutput -Force | Select-Object -First 1) {
        throw "Refusing to reuse a non-empty output directory: $resolvedOutput"
    }
}
else {
    New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
}
$resultPath = Join-Path $resolvedOutput 'benchmark.json'
$stdoutPath = Join-Path $resolvedOutput 'stdout.log'
$stderrPath = Join-Path $resolvedOutput 'stderr.log'
$telemetryPath = Join-Path $resolvedOutput 'gpu_telemetry.csv'
$guardPath = Join-Path $resolvedOutput 'guard_report.json'
$eventsPath = Join-Path $resolvedOutput 'system_events.json'
$detailsPath = Join-Path $resolvedOutput 'details.jsonl'

if (Test-Path -LiteralPath $resultPath) {
    throw "Refusing to overwrite an existing benchmark result: $resultPath"
}
if ($InferenceMode -eq 'full_plus_4_corner_crops') {
    if (-not $resolvedSelectionContract) {
        throw 'Crop/tiling comparison requires a hash-bound selection contract.'
    }
}

'sample_utc,temperature_c,utilization_percent,memory_used_mb,power_draw_w' |
    Set-Content -LiteralPath $telemetryPath -Encoding UTF8

$preflightRaw = & $nvidiaSmi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>$null
if ($LASTEXITCODE -ne 0 -or -not $preflightRaw) {
    throw 'GPU telemetry preflight failed before the benchmark was started.'
}
$preflightLine = @($preflightRaw)[0]
$preflightFields = @($preflightLine -split ',' | ForEach-Object { $_.Trim() })
if ($preflightFields.Count -lt 4) {
    throw "GPU telemetry preflight returned an unexpected row: $preflightLine"
}
try {
    $preflightCulture = [System.Globalization.CultureInfo]::InvariantCulture
    [void][double]::Parse($preflightFields[0], $preflightCulture)
    [void][double]::Parse($preflightFields[1], $preflightCulture)
    [void][double]::Parse($preflightFields[2], $preflightCulture)
    [void][double]::Parse($preflightFields[3], $preflightCulture)
}
catch {
    throw "GPU telemetry preflight contained a non-numeric value: $preflightLine"
}

$arguments = @(
    $benchmark,
    '--dataset-root', $resolvedDataset,
    '--weights', $resolvedWeights,
    '--output', $resultPath,
    '--details-output', $detailsPath,
    '--max-frames', $MaxFrames.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--selection', $Selection,
    '--batch', $Batch.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--imgsz', $ImageSize.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--no-half',
    '--memory-fraction', $MemoryFraction.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--inter-batch-delay-ms', $InterBatchDelayMs.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--inference-mode', $InferenceMode,
    '--inter-view-delay-ms', $InterViewDelayMs.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
if ($resolvedSelectionContract) {
    $arguments += @('--selection-contract', $resolvedSelectionContract)
}

$start = Get-Date
$startUtc = $start.ToUniversalTime()
$stopReason = $null
$monitorFailures = 0
$samples = 0
$maxObservedTemperatureC = $null
$maxObservedUtilizationPercent = $null
$maxObservedMemoryMb = $null
$maxObservedPowerW = $null
$exitCode = $null
$process = $null
$stdoutTask = $null
$stderrTask = $null

try {
    $processInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $processInfo.FileName = $resolvedPython
    $processInfo.WorkingDirectory = $repoRoot
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.Environment['YOLO_CONFIG_DIR'] = $resolvedOutput
    foreach ($argument in $arguments) {
        [void]$processInfo.ArgumentList.Add([string]$argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $processInfo
    if (-not $process.Start()) {
        throw 'Failed to start the guarded benchmark process.'
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    while (-not $process.HasExited) {
        Start-Sleep -Seconds 1
        $elapsed = ((Get-Date) - $start).TotalSeconds
        if ($elapsed -ge $TimeoutSeconds) {
            $stopReason = 'timeout'
            $process.Kill($true)
            break
        }

        $sampleUtc = (Get-Date).ToUniversalTime().ToString('o')
        $raw = & $nvidiaSmi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) {
            $monitorFailures += 1
            if ($monitorFailures -ge 3) {
                $stopReason = 'gpu_monitor_unavailable'
                $process.Kill($true)
                break
            }
            continue
        }

        $monitorFailures = 0
        $rawLine = @($raw)[0]
        $fields = @($rawLine -split ',' | ForEach-Object { $_.Trim() })
        if ($fields.Count -lt 4) {
            continue
        }
        $culture = [System.Globalization.CultureInfo]::InvariantCulture
        $temperature = [double]::Parse($fields[0], $culture)
        $utilization = [double]::Parse($fields[1], $culture)
        $memory = [double]::Parse($fields[2], $culture)
        $power = [double]::Parse($fields[3], $culture)
        "$sampleUtc,$temperature,$utilization,$memory,$power" |
            Add-Content -LiteralPath $telemetryPath -Encoding UTF8
        $samples += 1
        $maxObservedTemperatureC = if ($null -eq $maxObservedTemperatureC) { $temperature } else { [Math]::Max($maxObservedTemperatureC, $temperature) }
        $maxObservedUtilizationPercent = if ($null -eq $maxObservedUtilizationPercent) { $utilization } else { [Math]::Max($maxObservedUtilizationPercent, $utilization) }
        $maxObservedMemoryMb = if ($null -eq $maxObservedMemoryMb) { $memory } else { [Math]::Max($maxObservedMemoryMb, $memory) }
        $maxObservedPowerW = if ($null -eq $maxObservedPowerW) { $power } else { [Math]::Max($maxObservedPowerW, $power) }
        Write-Host ("GPU {0:N0}C util={1:N0}% memory={2:N0}MB power={3:N1}W" -f $temperature, $utilization, $memory, $power)

        if ($temperature -ge $MaxTemperatureC) {
            $stopReason = 'temperature_limit'
            $process.Kill($true)
            break
        }
    }

    $process.WaitForExit()
    $exitCode = $process.ExitCode
    $stdoutTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $stdoutPath -Encoding UTF8
    $stderrTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $stderrPath -Encoding UTF8
}
finally {
    if ($null -ne $process -and -not $process.HasExited) {
        $process.Kill($true)
        $process.WaitForExit()
    }
    if ($null -ne $stdoutTask -and -not (Test-Path -LiteralPath $stdoutPath)) {
        $stdoutTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $stdoutPath -Encoding UTF8
    }
    if ($null -ne $stderrTask -and -not (Test-Path -LiteralPath $stderrPath)) {
        $stderrTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $stderrPath -Encoding UTF8
    }
}

$end = Get-Date
$eventProviders = @(
    'Display',
    'nvlddmkm',
    'Microsoft-Windows-WHEA-Logger',
    'Microsoft-Windows-WER-SystemErrorReporting'
)
$events = @(
    Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $start } -ErrorAction SilentlyContinue |
        Where-Object { $_.ProviderName -in $eventProviders } |
        Select-Object TimeCreated, ProviderName, Id, LevelDisplayName, Message
)
ConvertTo-Json -InputObject $events -Depth 4 | Set-Content -LiteralPath $eventsPath -Encoding UTF8

$guard = [ordered]@{
    format = 'blindassist_guarded_gpu_run_v1'
    started_utc = $startUtc.ToString('o')
    ended_utc = $end.ToUniversalTime().ToString('o')
    elapsed_s = ($end - $start).TotalSeconds
    process_id = if ($null -ne $process) { $process.Id } else { $null }
    exit_code = $exitCode
    stop_reason = $stopReason
    monitor_samples = $samples
    limits = [ordered]@{
        max_temperature_c = $MaxTemperatureC
        timeout_s = $TimeoutSeconds
        memory_fraction = $MemoryFraction
        max_frames = $MaxFrames
        batch = $Batch
        image_size = $ImageSize
        inter_batch_delay_ms = $InterBatchDelayMs
        inter_view_delay_ms = $InterViewDelayMs
        inference_mode = $InferenceMode
        selection_contract = $resolvedSelectionContract
        views_per_frame_expected = if ($InferenceMode -eq 'full_plus_4_corner_crops') { 5 } else { 1 }
        max_inference_views = if ($InferenceMode -eq 'full_plus_4_corner_crops') { 5 * $MaxFrames } else { $MaxFrames }
    }
    observed = [ordered]@{
        max_temperature_c = $maxObservedTemperatureC
        max_utilization_percent = $maxObservedUtilizationPercent
        max_memory_used_mb = $maxObservedMemoryMb
        max_power_draw_w = $maxObservedPowerW
        relevant_system_events = $events.Count
    }
    artifacts = [ordered]@{
        benchmark = $resultPath
        stdout = $stdoutPath
        stderr = $stderrPath
        gpu_telemetry = $telemetryPath
        system_events = $eventsPath
        details = $detailsPath
    }
}
$guard | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $guardPath -Encoding UTF8

if ($stopReason) {
    throw "Guarded benchmark stopped: $stopReason"
}
if ($exitCode -ne 0) {
    throw "Benchmark process exited with code $exitCode; see $stderrPath"
}
if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
    throw "Benchmark did not write its expected result: $resultPath"
}
if (-not (Test-Path -LiteralPath $detailsPath -PathType Leaf)) {
    throw "Benchmark did not write its expected frame details: $detailsPath"
}
$benchmarkReport = Get-Content -LiteralPath $resultPath -Encoding UTF8 -Raw | ConvertFrom-Json
$expectedViews = if ($InferenceMode -eq 'full_plus_4_corner_crops') { 5 * $MaxFrames } else { $MaxFrames }
if ($benchmarkReport.compute_backend.inference_views -ne $expectedViews) {
    throw "Benchmark inference view count mismatch: expected $expectedViews, got $($benchmarkReport.compute_backend.inference_views)"
}
if ($benchmarkReport.model.inference_mode -ne $InferenceMode) {
    throw "Benchmark inference mode mismatch: expected $InferenceMode, got $($benchmarkReport.model.inference_mode)"
}
if ($events.Count -gt 0) {
    throw "Relevant system events were recorded during the run; see $eventsPath"
}

Get-Content -LiteralPath $resultPath -Encoding UTF8 -Raw
