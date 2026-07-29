[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = (
    "E:\codex-tools\tools\python-3.11-blindassist\python.exe"
)
$guard = Join-Path $PSScriptRoot "run_guarded_host_research.ps1"
$caseId = "guarded-run-test-{0}" -f ([guid]::NewGuid().ToString("N"))
$caseRelative = "artifacts.local/tmp/$caseId"
$caseDirectory = Join-Path $repoRoot $caseRelative
New-Item -ItemType Directory -Path $caseDirectory -Force |
    Out-Null
$monitorDirectories = [System.Collections.Generic.List[string]]::new()

try {
    $runner = Join-Path $caseDirectory "runner.py"
    $runnerSource = @'
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--progress", type=Path, required=True)
parser.add_argument("--success", type=Path, required=True)
parser.add_argument("--workers", type=int, required=True)
parser.add_argument("--progress-status", default="complete")
args = parser.parse_args()
args.progress.parent.mkdir(parents=True, exist_ok=True)
args.progress.write_text(
    json.dumps(
        {
            "phase": "producer",
            "completed_units": 2,
            "total_units": 2,
            "throughput": 10.0,
            "eta_seconds": 0,
            "last_progress_at": (
                __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat()
            ),
            "status": args.progress_status,
        }
    ),
    encoding="utf-8",
)
args.success.write_text(
    json.dumps({"status": "complete", "workers": args.workers}),
    encoding="utf-8",
)
'@
    Set-Content `
        -LiteralPath $runner `
        -Value $runnerSource `
        -Encoding UTF8
    $runnerHash = (Get-FileHash -LiteralPath $runner -Algorithm SHA256).
        Hash.ToLowerInvariant()
    $runnerRelative = "$caseRelative/runner.py"
    $progressRelative = "$caseRelative/progress.json"
    $successRelative = "$caseRelative/success.json"
    $failureRelative = "$caseRelative/failure.json"
    $receiptPath = Join-Path $caseDirectory "preflight.json"
    $receipt = [ordered]@{
        schema_version = "blindassist.host_research_preflight.v1"
        task_id = "TEST-GUARDED-HOST-RUN"
        execution_class = "long"
        implementation = [ordered]@{
            script = $runnerRelative
            sha256 = $runnerHash
        }
        workload = [ordered]@{
            class = "cpu_data_parallel"
            real_data_mechanics_match = $true
            input_identity = "generated-fixture:$("a" * 64)"
        }
        pilot = [ordered]@{
            representative_units = 2
            wall_seconds = 0.2
            projected_full_units = 2
            projected_full_wall_seconds = 0.2
            maximum_expected_wall_seconds = 0.5
            same_access_mechanics = $true
            output_equivalence = "PASS"
            progress_samples = 2
        }
        scheduler = [ordered]@{
            backend = "cpu_process_pool"
            workers = 2
            reason = "Integration fixture for guarded worker injection."
            comparison_performed = $true
            scientific_parameters_unchanged = $true
            estimated_gib_per_worker = 0.05
            reserve_memory_gib = 0.5
            requires_ac_power = $false
            inject_workers = $true
        }
        progress = [ordered]@{
            path = $progressRelative
            fields = @(
                "phase",
                "completed_units",
                "total_units",
                "throughput",
                "eta_seconds",
                "last_progress_at",
                "status"
            )
            update_interval_seconds = 1
            verified_in_pilot = $true
        }
        terminal = [ordered]@{
            success_path = $successRelative
            failure_path = $failureRelative
        }
    }
    Set-Content `
        -LiteralPath $receiptPath `
        -Value ($receipt | ConvertTo-Json -Depth 8) `
        -Encoding UTF8

    $guardOutput = & $guard `
        -PreflightReceipt $receiptPath `
        -Script $runner `
        -RepoRoot $repoRoot `
        -Python $python `
        -MonitorPollSeconds 1 `
        -RunnerArguments @(
            "--progress",
            (Join-Path $repoRoot $progressRelative),
            "--success",
            (Join-Path $repoRoot $successRelative)
        )
    $guardResult = $guardOutput |
        Select-Object -Last 1 |
        ConvertFrom-Json
    $monitorDirectories.Add([string]$guardResult.monitor_directory)
    Write-Output ($guardResult | ConvertTo-Json -Compress)
    if ($LASTEXITCODE -ne 0) {
        throw "Guarded launcher returned $LASTEXITCODE"
    }
    $success = Get-Content `
        -LiteralPath (Join-Path $repoRoot $successRelative) `
        -Raw `
        -Encoding UTF8 |
        ConvertFrom-Json
    if ($success.status -ne "complete" -or $success.workers -ne 2) {
        throw "Guarded launcher did not preserve success/worker contract."
    }

    Remove-Item `
        -LiteralPath (Join-Path $repoRoot $successRelative) `
        -Force
    Remove-Item `
        -LiteralPath (Join-Path $repoRoot $progressRelative) `
        -Force
    $runningOutput = & $guard `
        -PreflightReceipt $receiptPath `
        -Script $runner `
        -RepoRoot $repoRoot `
        -Python $python `
        -MonitorPollSeconds 1 `
        -RunnerArguments @(
            "--progress",
            (Join-Path $repoRoot $progressRelative),
            "--success",
            (Join-Path $repoRoot $successRelative),
            "--progress-status",
            "running"
        )
    $runningExitCode = $LASTEXITCODE
    $runningResult = $runningOutput |
        Select-Object -Last 1 |
        ConvertFrom-Json
    $monitorDirectories.Add([string]$runningResult.monitor_directory)
    if (
        $runningExitCode -ne 3 -or
        $runningResult.status -ne "PROGRESS_CONTRACT_VIOLATION" -or
        $runningResult.progress_contract_valid -ne $false
    ) {
        throw "Incomplete progress was not rejected."
    }

    Remove-Item `
        -LiteralPath (Join-Path $repoRoot $successRelative) `
        -Force
    $staleRejected = $false
    try {
        & $guard `
            -PreflightReceipt $receiptPath `
            -Script $runner `
            -RepoRoot $repoRoot `
            -Python $python `
            -MonitorPollSeconds 1 `
            -RunnerArguments @(
                "--progress",
                (Join-Path $repoRoot $progressRelative),
                "--success",
                (Join-Path $repoRoot $successRelative)
            )
    } catch {
        $staleRejected = $_.Exception.Message -like (
            "PROGRESS_PATH_ALREADY_EXISTS*"
        )
    }
    if (-not $staleRejected) {
        throw "Pre-existing progress was not rejected."
    }
    Remove-Item `
        -LiteralPath (Join-Path $repoRoot $progressRelative) `
        -Force

    $invalidReceipt = $receipt.PSObject.Copy()
    $invalidReceipt.progress = $receipt.progress.PSObject.Copy()
    $invalidReceipt.progress.verified_in_pilot = $false
    $invalidPath = Join-Path $caseDirectory "invalid-preflight.json"
    Set-Content `
        -LiteralPath $invalidPath `
        -Value ($invalidReceipt | ConvertTo-Json -Depth 8) `
        -Encoding UTF8
    $rejected = $false
    try {
        & $guard `
            -PreflightReceipt $invalidPath `
            -Script $runner `
            -RepoRoot $repoRoot `
            -Python $python `
            -MonitorPollSeconds 1 `
            -RunnerArguments @(
                "--progress",
                (Join-Path $repoRoot $progressRelative),
                "--success",
                (Join-Path $repoRoot $successRelative)
            )
    } catch {
        $rejected = $_.Exception.Message -like (
            "PERFORMANCE_NOT_QUALIFIED*"
        )
    }
    if (-not $rejected) {
        throw "Invalid preflight receipt was not rejected."
    }
    if (Test-Path -LiteralPath (Join-Path $repoRoot $successRelative)) {
        throw "Runner started despite invalid preflight receipt."
    }
    Write-Output "PASS: guarded host research launcher"
} finally {
    $resolvedCase = [IO.Path]::GetFullPath($caseDirectory)
    $resolvedTmp = [IO.Path]::GetFullPath(
        (Join-Path $repoRoot "artifacts.local\tmp")
    )
    if ($resolvedCase.StartsWith(
        $resolvedTmp + [IO.Path]::DirectorySeparatorChar
    )) {
        Remove-Item -LiteralPath $resolvedCase -Recurse -Force
    }
    foreach ($monitorDirectory in $monitorDirectories) {
        if ([string]::IsNullOrWhiteSpace($monitorDirectory)) {
            continue
        }
        $resolvedMonitor = [IO.Path]::GetFullPath($monitorDirectory)
        $monitorRoot = [IO.Path]::GetFullPath(
            (
                Join-Path `
                    $repoRoot `
                    "artifacts.local\evidence\host_guarded_runs"
            )
        )
        if ($resolvedMonitor.StartsWith(
            $monitorRoot + [IO.Path]::DirectorySeparatorChar
        )) {
            Remove-Item `
                -LiteralPath $resolvedMonitor `
                -Recurse `
                -Force `
                -ErrorAction SilentlyContinue
        }
    }
}
