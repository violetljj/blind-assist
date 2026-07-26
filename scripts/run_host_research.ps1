[CmdletBinding()]
param(
    [ValidateSet("interactive", "balanced", "throughput")]
    [string]$Profile = "balanced",

    [Parameter(Mandatory = $true)]
    [string]$Script,

    [string]$Python = "E:\codex-tools\bin\blindassist-python.cmd",

    [ValidateRange(0, 256)]
    [int]$Workers = 0,

    [ValidateRange(0.05, 64.0)]
    [double]$EstimatedGiBPerWorker = 0.30,

    [ValidateRange(0.5, 128.0)]
    [double]$ReserveMemoryGiB = 2.5,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Script -PathType Leaf)) {
    throw "Research script does not exist: $Script"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python launcher does not exist: $Python"
}
if ($RunnerArguments -contains "--workers") {
    throw "Pass worker count through -Workers or -Profile, not --workers in RunnerArguments."
}

$logicalProcessors = [Environment]::ProcessorCount
$profileTarget = switch ($Profile) {
    "interactive" { 8 }
    "balanced" { 12 }
    "throughput" { 16 }
}
$processorCap = [Math]::Max(1, $logicalProcessors - 2)
$requestedWorkers = if ($Workers -gt 0) { $Workers } else { $profileTarget }

$os = Get-CimInstance Win32_OperatingSystem
$availableGiB = [double]$os.FreePhysicalMemory / 1MB
$memoryBudgetGiB = [Math]::Max(0.0, $availableGiB - $ReserveMemoryGiB)
$memoryCap = [Math]::Max(
    1,
    [int][Math]::Floor($memoryBudgetGiB / $EstimatedGiBPerWorker)
)
$resolvedWorkers = [Math]::Max(
    1,
    [Math]::Min($requestedWorkers, [Math]::Min($processorCap, $memoryCap))
)

$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
$onAcPower = $null -eq $battery -or $battery.BatteryStatus -in 2, 3, 6, 7, 8, 9, 11
$summary = [ordered]@{
    scope = "HOST_RESEARCH_ONLY"
    profile = $Profile
    requested_workers = $requestedWorkers
    resolved_workers = $resolvedWorkers
    logical_processors = $logicalProcessors
    available_memory_gib = [Math]::Round($availableGiB, 2)
    reserve_memory_gib = $ReserveMemoryGiB
    estimated_gib_per_worker = $EstimatedGiBPerWorker
    on_ac_power = $onAcPower
    nested_numeric_threads = 1
    script = (Resolve-Path -LiteralPath $Script).Path
}
Write-Host ($summary | ConvertTo-Json -Compress)
if (-not $onAcPower) {
    Write-Warning "Host research is running on battery; sustained throughput may be power-limited."
}
if ($resolvedWorkers -lt $requestedWorkers) {
    Write-Warning (
        "Workers reduced from {0} to {1} by CPU/memory guard." -f
        $requestedWorkers,
        $resolvedWorkers
    )
}

$threadVariables = @(
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS"
)
$previousValues = @{}
foreach ($name in $threadVariables) {
    $previousValues[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    [Environment]::SetEnvironmentVariable($name, "1", "Process")
}

try {
    & $Python $Script @RunnerArguments --workers $resolvedWorkers
    $runnerExitCode = $LASTEXITCODE
} finally {
    foreach ($name in $threadVariables) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $previousValues[$name],
            "Process"
        )
    }
}

if ($null -eq $runnerExitCode) {
    $runnerExitCode = 0
}
exit $runnerExitCode
