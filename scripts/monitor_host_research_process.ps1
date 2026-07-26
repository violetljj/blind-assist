[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$ProcessId,

    [Parameter(Mandatory = $true)]
    [string]$EvidenceDirectory,

    [string]$AttemptBaseName = "formal_run_r0",

    [string]$MonitorDirectory = (
        "artifacts.local\evidence\host_process_monitors"
    ),

    [ValidateRange(1, 3600)]
    [int]$PollSeconds = 30,

    [ValidateRange(2, 100)]
    [int]$StallWindows = 3,

    [ValidateRange(0, 1000000)]
    [int]$MaxSamples = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-Phase {
    param(
        [bool]$ProcessExists,
        [string]$EvidencePath,
        [string]$BaseName
    )

    if (Test-Path -LiteralPath (
        Join-Path $EvidencePath $BaseName
    )) {
        return "complete"
    }
    if (Test-Path -LiteralPath (
        Join-Path $EvidencePath "$BaseName.failure.json"
    )) {
        return "failed"
    }
    $temporary = Get-ChildItem `
        -LiteralPath $EvidencePath `
        -Directory `
        -Force `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like ".$BaseName.*.tmp" } |
        Select-Object -First 1
    if ($null -ne $temporary) {
        return "validator_or_finalization"
    }
    if (-not $ProcessExists) {
        return "exited_without_receipt"
    }
    if (Test-Path -LiteralPath (
        Join-Path $EvidencePath "$BaseName.claim.json"
    )) {
        return "producer"
    }
    return "pre_claim"
}

$resolvedEvidence = (
    Resolve-Path -LiteralPath $EvidenceDirectory
).Path
$resolvedMonitor = [IO.Path]::GetFullPath(
    (Join-Path (Get-Location) $MonitorDirectory)
)
$attemptDirectory = Join-Path $resolvedMonitor (
    "{0}-pid-{1}" -f $AttemptBaseName, $ProcessId
)
New-Item -ItemType Directory -Path $attemptDirectory -Force |
    Out-Null

$historyPath = Join-Path $attemptDirectory "monitor.jsonl"
$latestPath = Join-Path $attemptDirectory "status.latest.json"
$previous = $null
$silentWindows = 0
$sampleCount = 0

while ($true) {
    $now = Get-Date
    $process = Get-CimInstance `
        -ClassName Win32_Process `
        -Filter "ProcessId=$ProcessId" `
        -ErrorAction SilentlyContinue
    $exists = $null -ne $process
    $phase = Get-Phase `
        -ProcessExists $exists `
        -EvidencePath $resolvedEvidence `
        -BaseName $AttemptBaseName

    $cpuSeconds = $null
    $readBytes = $null
    $writeBytes = $null
    $privateBytes = $null
    if ($exists) {
        $cpuSeconds = (
            [double]$process.KernelModeTime +
            [double]$process.UserModeTime
        ) / 1e7
        $readBytes = [double]$process.ReadTransferCount
        $writeBytes = [double]$process.WriteTransferCount
        $privateBytes = [double]$process.PrivatePageCount
    }

    $wallSeconds = $null
    $cpuCoreEquivalent = $null
    $readMiBPerSecond = $null
    $writeMiBPerSecond = $null
    if ($exists -and $null -ne $previous) {
        $wallSeconds = (
            $now - [datetime]$previous.timestamp
        ).TotalSeconds
        if ($wallSeconds -gt 0) {
            $cpuCoreEquivalent = [Math]::Max(
                0.0,
                ($cpuSeconds - [double]$previous.cpu_seconds) /
                    $wallSeconds
            )
            $readMiBPerSecond = [Math]::Max(
                0.0,
                ($readBytes - [double]$previous.read_bytes) /
                    1MB /
                    $wallSeconds
            )
            $writeMiBPerSecond = [Math]::Max(
                0.0,
                ($writeBytes - [double]$previous.write_bytes) /
                    1MB /
                    $wallSeconds
            )
        }
    }

    $health = if (-not $exists) {
        "not_running"
    } elseif ($null -eq $previous) {
        "sampling"
    } elseif (
        $cpuCoreEquivalent -ge 0.05 -or
        $readMiBPerSecond -ge 1.0 -or
        $writeMiBPerSecond -ge 1.0
    ) {
        $silentWindows = 0
        "running_progress"
    } else {
        $silentWindows += 1
        if ($silentWindows -ge $StallWindows) {
            "possible_stall"
        } else {
            "low_activity"
        }
    }

    $bottleneckHint = "insufficient_sample"
    $actionHint = "Collect another telemetry window."
    if ($phase -in @("complete", "failed", "exited_without_receipt")) {
        $bottleneckHint = "terminal"
        $actionHint = "Inspect the canonical output or failure state."
    } elseif ($health -eq "possible_stall") {
        $bottleneckHint = "possible_stall"
        $actionHint = (
            "Inspect the process stack and dependencies; do not restart an " +
            "irreversible formal attempt without protocol authority."
        )
    } elseif ($null -ne $cpuCoreEquivalent) {
        if (
            $cpuCoreEquivalent -le 1.2 -and
            $readMiBPerSecond -ge 100.0
        ) {
            $bottleneckHint = "serial_archive_or_decode_io"
            $actionHint = (
                "For the next implementation revision, materialize a " +
                "hash-bound cache before the formal claim, then benchmark " +
                "ordered parallel workers."
            )
        } elseif (
            $cpuCoreEquivalent -ge 0.7 -and
            $cpuCoreEquivalent -le 1.2 -and
            $readMiBPerSecond -lt 10.0
        ) {
            $bottleneckHint = "serial_cpu"
            $actionHint = (
                "Profile Python loops and native calls; vectorize or use an " +
                "ordered process pool in the next implementation revision."
            )
        } elseif ($cpuCoreEquivalent -gt 1.2) {
            $bottleneckHint = "parallel_cpu"
            $actionHint = (
                "Compare throughput scaling and memory headroom with the " +
                "host research worker profiles."
            )
        } elseif (
            $readMiBPerSecond -ge 100.0 -and
            $cpuCoreEquivalent -lt 0.7
        ) {
            $bottleneckHint = "io_heavy"
            $actionHint = (
                "Reduce repeated reads and benchmark bounded I/O concurrency."
            )
        } elseif ($health -eq "running_progress") {
            $bottleneckHint = "active_mixed_or_low_parallelism"
            $actionHint = (
                "Use a bounded pilot and profiler before choosing CPU, GPU, " +
                "or I/O concurrency."
            )
        }
    }

    $record = [ordered]@{
        schema_version = "blindassist.host_process_monitor.v1"
        timestamp = $now.ToUniversalTime().ToString("o")
        process_id = $ProcessId
        process_exists = $exists
        phase = $phase
        health = $health
        cpu_seconds = if ($null -eq $cpuSeconds) {
            $null
        } else {
            [Math]::Round($cpuSeconds, 3)
        }
        cpu_core_equivalent = if ($null -eq $cpuCoreEquivalent) {
            $null
        } else {
            [Math]::Round($cpuCoreEquivalent, 3)
        }
        read_gib_total = if ($null -eq $readBytes) {
            $null
        } else {
            [Math]::Round($readBytes / 1GB, 3)
        }
        read_mib_per_second = if ($null -eq $readMiBPerSecond) {
            $null
        } else {
            [Math]::Round($readMiBPerSecond, 3)
        }
        write_mib_per_second = if ($null -eq $writeMiBPerSecond) {
            $null
        } else {
            [Math]::Round($writeMiBPerSecond, 3)
        }
        private_memory_mib = if ($null -eq $privateBytes) {
            $null
        } else {
            [Math]::Round($privateBytes / 1MB, 1)
        }
        consecutive_low_activity_windows = $silentWindows
        bottleneck_hint = $bottleneckHint
        action_hint = $actionHint
        progress_fraction = $null
        eta_seconds = $null
        progress_note = (
            "Exact progress and ETA are unavailable because the observed " +
            "runner does not publish completed/total units."
        )
        evidence_directory = $resolvedEvidence
    }
    $json = $record | ConvertTo-Json -Compress
    Add-Content -LiteralPath $historyPath -Value $json -Encoding UTF8
    $latestTemporary = "$latestPath.$PID.tmp"
    Set-Content `
        -LiteralPath $latestTemporary `
        -Value ($record | ConvertTo-Json) `
        -Encoding UTF8
    Move-Item `
        -LiteralPath $latestTemporary `
        -Destination $latestPath `
        -Force
    Write-Output $json

    $sampleCount += 1
    if (
        $phase -in @(
            "complete",
            "failed",
            "exited_without_receipt"
        ) -or
        ($MaxSamples -gt 0 -and $sampleCount -ge $MaxSamples)
    ) {
        break
    }

    if ($exists) {
        $previous = [pscustomobject]@{
            timestamp = $now
            cpu_seconds = $cpuSeconds
            read_bytes = $readBytes
            write_bytes = $writeBytes
        }
    } else {
        $previous = $null
    }
    Start-Sleep -Seconds $PollSeconds
}
