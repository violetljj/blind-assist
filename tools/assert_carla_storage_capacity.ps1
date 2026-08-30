[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('Guard', 'Acquire', 'Check', 'Release')]
    [string]$Action = 'Guard',
    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$ExperimentsRoot = '',
    [string]$Policy = 'research/active/dtr-r0/carla/carla_storage_policy.json',
    [ValidateRange(0, [long]::MaxValue)]
    [long]$ReservationBytes = 0,
    [string]$OutputRoot = '',
    [string]$LeaseLabel = '',
    [string]$LeaseToken = '',
    [ValidateRange(1, [int]::MaxValue)]
    [int]$OwnerPid = $PID
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

function Resolve-TaskPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

$carlaRootPath = Resolve-TaskPath -Value $CarlaRoot
$pythonPath = Resolve-TaskPath -Value $CarlaPython
$policyPath = Resolve-TaskPath -Value $Policy
$storageToolPath = Join-Path `
    $repoRoot `
    'research/active/dtr-r0/carla/carla_storage.py'
$experimentsPath = if ([string]::IsNullOrWhiteSpace($ExperimentsRoot)) {
    Join-Path $carlaRootPath 'experiments'
} else {
    Resolve-TaskPath -Value $ExperimentsRoot
}

foreach ($required in @($pythonPath, $policyPath, $storageToolPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "CARLA storage guard dependency is unavailable: $required"
    }
}
if (-not (Test-Path -LiteralPath $carlaRootPath -PathType Container)) {
    throw "CARLA root is unavailable: $carlaRootPath"
}
if (-not (Test-Path -LiteralPath $experimentsPath -PathType Container)) {
    [IO.Directory]::CreateDirectory($experimentsPath) | Out-Null
}

$arguments = [Collections.Generic.List[string]]::new()
$command = switch ($Action) {
    'Guard' { 'guard' }
    'Acquire' { 'lease-acquire' }
    'Check' { 'lease-check' }
    'Release' { 'lease-release' }
}
foreach ($value in @($storageToolPath, $command, '--root', $experimentsPath)) {
    $arguments.Add([string]$value)
}
$requiresPolicy = $Action -in @('Guard', 'Acquire', 'Check')
if ($requiresPolicy) {
    $arguments.Add('--policy')
    $arguments.Add($policyPath)
}
if ($Action -eq 'Guard' -and $ReservationBytes -gt 0) {
    $arguments.Add('--reservation-bytes')
    $arguments.Add([string]$ReservationBytes)
}
if ($Action -eq 'Acquire') {
    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        throw 'CARLA storage lease acquisition requires -OutputRoot.'
    }
    if ([string]::IsNullOrWhiteSpace($LeaseLabel)) {
        throw 'CARLA storage lease acquisition requires -LeaseLabel.'
    }
    foreach ($value in @(
        '--reservation-bytes', [string]$ReservationBytes,
        '--owner-pid', [string]$OwnerPid,
        '--label', $LeaseLabel,
        '--output-root', (Resolve-TaskPath -Value $OutputRoot)
    )) {
        $arguments.Add([string]$value)
    }
}
if ($Action -in @('Check', 'Release')) {
    if ([string]::IsNullOrWhiteSpace($LeaseToken)) {
        throw "CARLA storage $Action requires -LeaseToken."
    }
    $arguments.Add('--lease-token')
    $arguments.Add($LeaseToken)
    if ($Action -eq 'Check' -and -not [string]::IsNullOrWhiteSpace($OutputRoot)) {
        $arguments.Add('--output-root')
        $arguments.Add((Resolve-TaskPath -Value $OutputRoot))
    }
}

$jsonText = (& $pythonPath @arguments | Out-String)
$exitCode = $LASTEXITCODE
try {
    $result = $jsonText | ConvertFrom-Json -Depth 100
}
catch {
    throw "CARLA storage guard returned invalid JSON (exit $exitCode): $jsonText"
}
if ($exitCode -ne 0) {
    throw "CARLA storage $Action failed with exit code $exitCode."
}
if ($Action -eq 'Guard' -and [string]$result.status -ne 'PASS') {
    $reasons = @($result.reasons) -join ','
    throw (
        "CARLA storage guard refused the run: reasons=$reasons " +
        "unique=$($result.accounting.unique_bytes) " +
        "projected=$($result.projected_unique_bytes) " +
        "cap=$($result.policy.maximum_experiment_unique_bytes) " +
        "projected_free=$($result.projected_volume_free_bytes) " +
        "free_floor=$($result.policy.minimum_volume_free_bytes)"
    )
}

if ($Action -eq 'Guard') {
    Write-Output (
        "CARLA_STORAGE PASS unique=$($result.accounting.unique_bytes) " +
        "projected=$($result.projected_unique_bytes) " +
        "cap=$($result.policy.maximum_experiment_unique_bytes) " +
        "projected_free=$($result.projected_volume_free_bytes)"
    )
}
else {
    Write-Output ($result | ConvertTo-Json -Compress -Depth 100)
}
