[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$ProcessId,

    [Parameter(Mandatory = $true)]
    [string]$EvidenceDirectory,

    [string]$AttemptBaseName = "formal_run_r0",

    [string]$ClaimPath = "",

    [string]$SuccessPath = "",

    [string]$FailurePath = "",

    [string]$ProgressPath = "",

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
        [string]$BaseName,
        [string]$ObservedClaim,
        [string]$ObservedSuccess,
        [string]$ObservedFailure,
        [string]$RunnerPhase
    )

    if (Test-Path -LiteralPath $ObservedSuccess) {
        return "complete"
    }
    if (Test-Path -LiteralPath $ObservedFailure) {
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
    if (-not [string]::IsNullOrWhiteSpace($RunnerPhase)) {
        return "runner_$RunnerPhase"
    }
    if (Test-Path -LiteralPath $ObservedClaim) {
        return "producer"
    }
    return "pre_claim"
}

function Get-RunnerProgressValue {
    param(
        [object]$Progress,
        [string]$Name
    )

    if ($null -eq $Progress) {
        return $null
    }
    $property = $Progress.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Get-ProcessTreeSnapshot {
    param(
        [int]$RootProcessId
    )

    $allProcesses = @(
        Get-CimInstance `
            -ClassName Win32_Process `
            -ErrorAction SilentlyContinue
    )
    $byProcessId = @{}
    $childrenByParent = @{}
    foreach ($item in $allProcesses) {
        $itemProcessId = [int]$item.ProcessId
        $parentProcessId = [int]$item.ParentProcessId
        $byProcessId[$itemProcessId] = $item
        if (-not $childrenByParent.ContainsKey($parentProcessId)) {
            $childrenByParent[$parentProcessId] = [Collections.Generic.List[int]]::new()
        }
        $childrenByParent[$parentProcessId].Add($itemProcessId)
    }

    if (-not $byProcessId.ContainsKey($RootProcessId)) {
        return @()
    }

    $pending = [Collections.Generic.Queue[int]]::new()
    $pending.Enqueue($RootProcessId)
    $visited = [Collections.Generic.HashSet[int]]::new()
    $result = [Collections.Generic.List[object]]::new()
    while ($pending.Count -gt 0) {
        $currentProcessId = $pending.Dequeue()
        if (-not $visited.Add($currentProcessId)) {
            continue
        }
        if (-not $byProcessId.ContainsKey($currentProcessId)) {
            continue
        }
        $result.Add($byProcessId[$currentProcessId])
        if ($childrenByParent.ContainsKey($currentProcessId)) {
            foreach ($childProcessId in $childrenByParent[$currentProcessId]) {
                $pending.Enqueue($childProcessId)
            }
        }
    }
    return @($result)
}

function Get-NvidiaGpuSnapshot {
    $empty = [ordered]@{
        gpu_utilization_percent = $null
        gpu_memory_used_mib = $null
        gpu_temperature_c = $null
        gpu_power_draw_w = $null
    }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return [pscustomobject]$empty
    }

    try {
        $output = @(& $nvidiaSmi.Source `
            --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw `
            --format=csv,noheader,nounits `
            2>$null)
        $nativeExitCode = $LASTEXITCODE
        $line = $output | Select-Object -First 1
        if (
            ($null -ne $nativeExitCode -and $nativeExitCode -ne 0) -or
            [string]::IsNullOrWhiteSpace($line)
        ) {
            return [pscustomobject]$empty
        }
        $values = @([string]$line -split ",")
        if ($values.Count -ne 4) {
            return [pscustomobject]$empty
        }
        $parsed = @()
        foreach ($value in $values) {
            $number = 0.0
            $ok = [double]::TryParse(
                $value.Trim(),
                [Globalization.NumberStyles]::Float,
                [Globalization.CultureInfo]::InvariantCulture,
                [ref]$number
            )
            if (-not $ok) {
                return [pscustomobject]$empty
            }
            $parsed += $number
        }
        return [pscustomobject][ordered]@{
            gpu_utilization_percent = $parsed[0]
            gpu_memory_used_mib = $parsed[1]
            gpu_temperature_c = $parsed[2]
            gpu_power_draw_w = $parsed[3]
        }
    } catch {
        return [pscustomobject]$empty
    }
}

$resolvedEvidence = [IO.Path]::GetFullPath(
    (Join-Path (Get-Location) $EvidenceDirectory)
)
$resolvedClaim = if ([string]::IsNullOrWhiteSpace($ClaimPath)) {
    Join-Path $resolvedEvidence "$AttemptBaseName.claim.json"
} else {
    [IO.Path]::GetFullPath((Join-Path (Get-Location) $ClaimPath))
}
$resolvedSuccess = if ([string]::IsNullOrWhiteSpace($SuccessPath)) {
    Join-Path $resolvedEvidence $AttemptBaseName
} else {
    [IO.Path]::GetFullPath((Join-Path (Get-Location) $SuccessPath))
}
$resolvedFailure = if ([string]::IsNullOrWhiteSpace($FailurePath)) {
    Join-Path $resolvedEvidence "$AttemptBaseName.failure.json"
} else {
    [IO.Path]::GetFullPath((Join-Path (Get-Location) $FailurePath))
}
$resolvedProgress = if ([string]::IsNullOrWhiteSpace($ProgressPath)) {
    ""
} else {
    [IO.Path]::GetFullPath((Join-Path (Get-Location) $ProgressPath))
}
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
    $processTree = @(Get-ProcessTreeSnapshot -RootProcessId $ProcessId)
    $exists = $processTree.Count -gt 0
    $childCount = [Math]::Max(0, $processTree.Count - 1)
    $runnerProgress = $null
    if (
        -not [string]::IsNullOrWhiteSpace($resolvedProgress) -and
        (Test-Path -LiteralPath $resolvedProgress -PathType Leaf)
    ) {
        try {
            $runnerProgress = Get-Content `
                -LiteralPath $resolvedProgress `
                -Raw `
                -Encoding UTF8 |
                ConvertFrom-Json
        } catch {
            $runnerProgress = $null
        }
    }
    $runnerPhaseValue = Get-RunnerProgressValue `
        -Progress $runnerProgress `
        -Name "phase"
    $runnerPhase = if ($null -ne $runnerPhaseValue) {
        [string]$runnerPhaseValue
    } else {
        ""
    }
    $phase = Get-Phase `
        -ProcessExists $exists `
        -EvidencePath $resolvedEvidence `
        -BaseName $AttemptBaseName `
        -ObservedClaim $resolvedClaim `
        -ObservedSuccess $resolvedSuccess `
        -ObservedFailure $resolvedFailure `
        -RunnerPhase $runnerPhase

    $cpuSeconds = $null
    $readBytes = $null
    $writeBytes = $null
    $privateBytes = $null
    if ($exists) {
        $cpuSeconds = 0.0
        $readBytes = 0.0
        $writeBytes = 0.0
        $privateBytes = 0.0
        foreach ($process in $processTree) {
            $cpuSeconds += (
                [double]$process.KernelModeTime +
                [double]$process.UserModeTime
            ) / 1e7
            $readBytes += [double]$process.ReadTransferCount
            $writeBytes += [double]$process.WriteTransferCount
            $privateBytes += [double]$process.PrivatePageCount
        }
    }
    $gpu = Get-NvidiaGpuSnapshot

    $wallSeconds = $null
    $cpuCoreEquivalent = $null
    $readMiBPerSecond = $null
    $writeMiBPerSecond = $null
    $completedValue = Get-RunnerProgressValue `
        -Progress $runnerProgress `
        -Name "completed_units"
    $totalValue = Get-RunnerProgressValue `
        -Progress $runnerProgress `
        -Name "total_units"
    $completedUnits = if ($null -ne $completedValue) {
        [double]$completedValue
    } else {
        $null
    }
    $totalUnits = if ($null -ne $totalValue) {
        [double]$totalValue
    } else {
        $null
    }
    $unitProgressed = (
        $null -ne $completedUnits -and
        $null -ne $previous -and
        $null -ne $previous.completed_units -and
        $completedUnits -gt [double]$previous.completed_units
    )
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
        $unitProgressed -or
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
        child_count = $childCount
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
        gpu_utilization_percent = $gpu.gpu_utilization_percent
        gpu_memory_used_mib = $gpu.gpu_memory_used_mib
        gpu_temperature_c = $gpu.gpu_temperature_c
        gpu_power_draw_w = $gpu.gpu_power_draw_w
        consecutive_low_activity_windows = $silentWindows
        bottleneck_hint = $bottleneckHint
        action_hint = $actionHint
        completed_units = $completedUnits
        total_units = $totalUnits
        progress_fraction = if (
            $null -ne $completedUnits -and
            $null -ne $totalUnits -and
            $totalUnits -gt 0
        ) {
            [Math]::Round($completedUnits / $totalUnits, 6)
        } else {
            $null
        }
        throughput = Get-RunnerProgressValue `
            -Progress $runnerProgress `
            -Name "throughput"
        eta_seconds = Get-RunnerProgressValue `
            -Progress $runnerProgress `
            -Name "eta_seconds"
        last_progress_at = Get-RunnerProgressValue `
            -Progress $runnerProgress `
            -Name "last_progress_at"
        progress_note = if ($null -ne $runnerProgress) {
            "Runner-published progress is available."
        } else {
            (
                "Exact progress and ETA are unavailable because the observed " +
                "runner does not publish completed/total units."
            )
        }
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
            completed_units = $completedUnits
        }
    } else {
        $previous = $null
    }
    Start-Sleep -Seconds $PollSeconds
}
