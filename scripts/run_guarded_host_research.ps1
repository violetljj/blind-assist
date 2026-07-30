[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PreflightReceipt,

    [Parameter(Mandatory = $true)]
    [string]$Script,

    [string]$RepoRoot = (Get-Location).Path,

    [string]$Python = (
        "E:\codex-tools\tools\python-3.11-blindassist\python.exe"
    ),

    [ValidateRange(1, 3600)]
    [int]$MonitorPollSeconds = 10,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedRepo = (Resolve-Path -LiteralPath $RepoRoot).Path
$resolvedReceipt = (Resolve-Path -LiteralPath $PreflightReceipt).Path
$resolvedScript = (Resolve-Path -LiteralPath $Script).Path
$resolvedPython = (Resolve-Path -LiteralPath $Python).Path
$validator = Join-Path `
    $resolvedRepo `
    "scripts\validate_host_research_preflight.py"
$monitor = Join-Path `
    $resolvedRepo `
    "scripts\monitor_host_research_process.ps1"

$validationOutput = & $resolvedPython `
    $validator `
    --repo-root $resolvedRepo `
    --receipt $resolvedReceipt `
    --expected-script $resolvedScript
$validationExitCode = $LASTEXITCODE
if ($validationExitCode -ne 0) {
    throw "PERFORMANCE_NOT_QUALIFIED: $validationOutput"
}

$receipt = Get-Content `
    -LiteralPath $resolvedReceipt `
    -Raw `
    -Encoding UTF8 |
    ConvertFrom-Json
$successPath = Join-Path $resolvedRepo $receipt.terminal.success_path
$failurePath = Join-Path $resolvedRepo $receipt.terminal.failure_path
$progressPath = Join-Path $resolvedRepo $receipt.progress.path
$evidenceDirectory = Split-Path -Parent $successPath

if (
    (Test-Path -LiteralPath $successPath) -or
    (Test-Path -LiteralPath $failurePath)
) {
    throw (
        "TERMINAL_PATH_ALREADY_EXISTS: success={0}; failure={1}" -f
        $successPath,
        $failurePath
    )
}
if (Test-Path -LiteralPath $progressPath) {
    throw "PROGRESS_PATH_ALREADY_EXISTS: $progressPath"
}
if ($receipt.execution_class -eq "formal") {
    $claimPath = Join-Path $resolvedRepo $receipt.formal.claim_path
    if (Test-Path -LiteralPath $claimPath) {
        throw "FORMAL_CLAIM_ALREADY_EXISTS: $claimPath"
    }
} else {
    $claimPath = ""
}

if (
    $receipt.scheduler.inject_workers -eq $true -and
    $RunnerArguments -contains "--workers"
) {
    throw (
        "Pass worker count through the preflight receipt, not RunnerArguments."
    )
}

$operatingSystem = Get-CimInstance Win32_OperatingSystem
$availableMemoryGiB = (
    [double]$operatingSystem.FreePhysicalMemory / 1MB
)
$requiredMemoryGiB = (
    [double]$receipt.scheduler.reserve_memory_gib +
    (
        [int]$receipt.scheduler.workers *
        [double]$receipt.scheduler.estimated_gib_per_worker
    )
)
if ($availableMemoryGiB -lt $requiredMemoryGiB) {
    throw (
        "HOST_CAPACITY_NOT_AVAILABLE: available RAM {0:N2} GiB; " +
        "receipt requires {1:N2} GiB." -f
        $availableMemoryGiB,
        $requiredMemoryGiB
    )
}

$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
$onAcPower = (
    $null -eq $battery -or
    $battery.BatteryStatus -in 2, 3, 6, 7, 8, 9, 11
)
if ($receipt.scheduler.requires_ac_power -eq $true -and -not $onAcPower) {
    throw "HOST_CAPACITY_NOT_AVAILABLE: AC power is required."
}

if ($receipt.scheduler.backend -in @("cuda", "mixed")) {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw "HOST_CAPACITY_NOT_AVAILABLE: nvidia-smi is unavailable."
    }
    $freeVramMiB = & $nvidiaSmi.Source `
        --query-gpu=memory.free `
        --format=csv,noheader,nounits |
        Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or $null -eq $freeVramMiB) {
        throw "HOST_CAPACITY_NOT_AVAILABLE: cannot query free VRAM."
    }
    $freeVramGiB = [double]$freeVramMiB / 1024.0
    $minimumVramGiB = [double](
        $receipt.scheduler.minimum_free_vram_gib
    )
    if ($freeVramGiB -lt $minimumVramGiB) {
        throw (
            "HOST_CAPACITY_NOT_AVAILABLE: free VRAM {0:N2} GiB; " +
            "receipt requires {1:N2} GiB." -f
            $freeVramGiB,
            $minimumVramGiB
        )
    }
}

$processInfo = [Diagnostics.ProcessStartInfo]::new()
$processInfo.FileName = $resolvedPython
$processInfo.WorkingDirectory = $resolvedRepo
$processInfo.UseShellExecute = $false
$processInfo.RedirectStandardOutput = $true
$processInfo.RedirectStandardError = $true
$processInfo.CreateNoWindow = $true
$processInfo.ArgumentList.Add($resolvedScript)
foreach ($argument in $RunnerArguments) {
    $processInfo.ArgumentList.Add($argument)
}
if ($receipt.scheduler.inject_workers -eq $true) {
    $processInfo.ArgumentList.Add("--workers")
    $processInfo.ArgumentList.Add(
        ([int]$receipt.scheduler.workers).ToString()
    )
}

if ($receipt.scheduler.backend -eq "cpu_process_pool") {
    foreach ($name in @(
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS"
    )) {
        $processInfo.Environment[$name] = "1"
    }
}

$process = [Diagnostics.Process]::new()
$process.StartInfo = $processInfo
$runnerStartedAtUtc = [DateTime]::UtcNow
if (-not $process.Start()) {
    throw "RUNNER_START_FAILED"
}
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()

$monitorParent = Join-Path `
    $resolvedRepo `
    "artifacts.local\evidence\host_guarded_runs"
$monitorRoot = Join-Path `
    $monitorParent `
    ("{0}-pid-{1}" -f $receipt.task_id, $process.Id)
New-Item -ItemType Directory -Path $monitorRoot -Force |
    Out-Null

$monitorInfo = [Diagnostics.ProcessStartInfo]::new()
$monitorInfo.FileName = (Get-Command pwsh).Source
$monitorInfo.WorkingDirectory = $resolvedRepo
$monitorInfo.UseShellExecute = $false
$monitorInfo.CreateNoWindow = $true
foreach ($argument in @(
    "-NoProfile",
    "-File",
    $monitor,
    "-ProcessId",
    $process.Id.ToString(),
    "-EvidenceDirectory",
    $evidenceDirectory,
    "-AttemptBaseName",
    $receipt.task_id,
    "-SuccessPath",
    $successPath,
    "-FailurePath",
    $failurePath,
    "-ProgressPath",
    $progressPath,
    "-MonitorDirectory",
    $monitorParent,
    "-PollSeconds",
    $MonitorPollSeconds.ToString()
)) {
    $monitorInfo.ArgumentList.Add([string]$argument)
}
if (-not [string]::IsNullOrWhiteSpace($claimPath)) {
    $monitorInfo.ArgumentList.Add("-ClaimPath")
    $monitorInfo.ArgumentList.Add($claimPath)
}
$monitorProcess = [Diagnostics.Process]::Start($monitorInfo)

$process.WaitForExit()
$stdout = $stdoutTask.GetAwaiter().GetResult()
$stderr = $stderrTask.GetAwaiter().GetResult()
Set-Content `
    -LiteralPath (Join-Path $monitorRoot "runner.stdout.log") `
    -Value $stdout `
    -Encoding UTF8
Set-Content `
    -LiteralPath (Join-Path $monitorRoot "runner.stderr.log") `
    -Value $stderr `
    -Encoding UTF8

if ($null -ne $monitorProcess) {
    [void]$monitorProcess.WaitForExit(
        [Math]::Max(5000, ($MonitorPollSeconds + 2) * 1000)
    )
    if (-not $monitorProcess.HasExited) {
        $monitorProcess.Kill()
        $monitorProcess.WaitForExit()
    }
}

$successExists = Test-Path -LiteralPath $successPath
$failureExists = Test-Path -LiteralPath $failurePath
$progressExists = Test-Path -LiteralPath $progressPath -PathType Leaf
$progressContractErrors = [System.Collections.Generic.List[string]]::new()
$progressRecord = $null
if ($progressExists) {
    try {
        $progressRecord = Get-Content `
            -LiteralPath $progressPath `
            -Raw `
            -Encoding UTF8 |
            ConvertFrom-Json
    } catch {
        $progressContractErrors.Add("progress is not valid JSON")
    }
}
if ($null -ne $progressRecord) {
    $progressProperties = @($progressRecord.PSObject.Properties.Name)
    foreach ($field in @($receipt.progress.fields)) {
        if ($field -notin $progressProperties) {
            $progressContractErrors.Add("missing field: $field")
        }
    }

    if ("status" -in $progressProperties) {
        $expectedProgressStatus = if ($failureExists) {
            "failed"
        } else {
            "complete"
        }
        if (
            [string]$progressRecord.status -ne
            $expectedProgressStatus
        ) {
            $progressContractErrors.Add(
                "progress.status must be $expectedProgressStatus"
            )
        }
    }

    $completedUnits = $null
    $totalUnits = $null
    if (
        "completed_units" -in $progressProperties -and
        "total_units" -in $progressProperties
    ) {
        try {
            $completedUnits = [double]$progressRecord.completed_units
            $totalUnits = [double]$progressRecord.total_units
            if ($failureExists) {
                if (
                    $completedUnits -lt 0 -or
                    $totalUnits -lt 0 -or
                    $completedUnits -gt $totalUnits
                ) {
                    $progressContractErrors.Add(
                        "failed progress requires 0 <= completed_units <= total_units"
                    )
                }
            } elseif (
                $completedUnits -lt 0 -or
                $totalUnits -lt 0 -or
                $completedUnits -ne $totalUnits
            ) {
                $progressContractErrors.Add(
                    "completed_units must equal total_units at completion"
                )
            }
        } catch {
            $progressContractErrors.Add(
                "completed_units and total_units must be numeric"
            )
        }
    }

    if ("last_progress_at" -in $progressProperties) {
        $lastProgressValue = $progressRecord.last_progress_at
        $lastProgressUtc = [DateTime]::MinValue
        $lastProgressParsed = $false
        if ($lastProgressValue -is [DateTime]) {
            $lastProgressUtc = (
                [DateTime]$lastProgressValue
            ).ToUniversalTime()
            $lastProgressParsed = $true
        } elseif ($lastProgressValue -is [DateTimeOffset]) {
            $lastProgressUtc = (
                [DateTimeOffset]$lastProgressValue
            ).UtcDateTime
            $lastProgressParsed = $true
        } else {
            $lastProgressAt = [DateTimeOffset]::MinValue
            $lastProgressParsed = [DateTimeOffset]::TryParse(
                [string]$lastProgressValue,
                [ref]$lastProgressAt
            )
            if ($lastProgressParsed) {
                $lastProgressUtc = $lastProgressAt.UtcDateTime
            }
        }
        if (-not $lastProgressParsed) {
            $progressContractErrors.Add(
                "last_progress_at must be an ISO-8601 timestamp"
            )
        } elseif (
            $lastProgressUtc -lt
            $runnerStartedAtUtc.AddSeconds(-2)
        ) {
            $progressContractErrors.Add(
                "last_progress_at predates this runner invocation"
            )
        }
    }

    $progressWriteUtc = (
        Get-Item -LiteralPath $progressPath
    ).LastWriteTimeUtc
    if ($progressWriteUtc -lt $runnerStartedAtUtc.AddSeconds(-2)) {
        $progressContractErrors.Add(
            "progress file predates this runner invocation"
        )
    }
}
$progressContractValid = (
    $progressExists -and
    $progressContractErrors.Count -eq 0
)
$terminalStatus = if (
    $process.ExitCode -eq 0 -and
    $successExists -and
    -not $failureExists -and
    $progressContractValid
) {
    "COMPLETE"
} elseif ($failureExists -and $progressContractValid) {
    "FAILED_WITH_RECEIPT"
} elseif (-not $progressContractValid) {
    "PROGRESS_CONTRACT_VIOLATION"
} elseif ($process.ExitCode -ne 0) {
    "FAILED_WITHOUT_RECEIPT"
} else {
    "TERMINAL_CONTRACT_VIOLATION"
}

$summary = [ordered]@{
    schema_version = "blindassist.host_guarded_run.v1"
    task_id = $receipt.task_id
    execution_class = $receipt.execution_class
    process_id = $process.Id
    exit_code = $process.ExitCode
    status = $terminalStatus
    success_path_exists = $successExists
    failure_path_exists = $failureExists
    progress_path_exists = $progressExists
    progress_contract_valid = $progressContractValid
    progress_contract_errors = @($progressContractErrors)
    preflight_receipt = $resolvedReceipt
    implementation_sha256 = $receipt.implementation.sha256
    monitor_directory = $monitorRoot
}
$summaryPath = Join-Path $monitorRoot "guarded_run_summary.json"
Set-Content `
    -LiteralPath $summaryPath `
    -Value ($summary | ConvertTo-Json) `
    -Encoding UTF8
Write-Output ($summary | ConvertTo-Json -Compress)

if ($terminalStatus -ne "COMPLETE") {
    exit 3
}
exit 0
