[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('check', 'run')]
    [string]$Command = 'check',
    [Parameter(Position = 1, Mandatory = $true)]
    [string]$Manifest,
    [ValidateSet('research-dtr-r0', 'research-l10-r0')]
    [string]$Profile = 'research-dtr-r0',
    [string]$SkyDiscoverRoot,
    [string]$SkyDiscoverPython,
    [string]$EvaluatorLauncher,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$LocalConfigPath = Join-Path $RepoRoot 'config/local.toml'

function Read-LocalConfig {
    $values = @{}
    if (-not (Test-Path -LiteralPath $LocalConfigPath -PathType Leaf)) { return $values }
    foreach ($line in Get-Content -LiteralPath $LocalConfigPath) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed.StartsWith('[')) { continue }
        if ($trimmed -match '^([A-Za-z0-9_-]+)\s*=\s*"(.*)"\s*$') {
            $values[$Matches[1]] = $Matches[2]
        }
    }
    return $values
}

function Resolve-LocalPath {
    param([string]$Value, [string]$Base = $RepoRoot)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Base $Value))
}

$local = Read-LocalConfig
$skyRootValue = $SkyDiscoverRoot
if ([string]::IsNullOrWhiteSpace($skyRootValue) -and $local.ContainsKey('skydiscover_root')) {
    $skyRootValue = $local['skydiscover_root']
}
if ([string]::IsNullOrWhiteSpace($skyRootValue)) {
    $skyRootValue = $env:SKYDISCOVER_ROOT
}
$skyRoot = Resolve-LocalPath $skyRootValue
if (-not $skyRoot -or -not (Test-Path -LiteralPath (Join-Path $skyRoot 'pyproject.toml') -PathType Leaf)) {
    throw 'SkyDiscover root is unavailable. Set skydiscover_root in config/local.toml, SKYDISCOVER_ROOT, or -SkyDiscoverRoot.'
}

$skyPythonValue = $SkyDiscoverPython
if ([string]::IsNullOrWhiteSpace($skyPythonValue) -and $local.ContainsKey('skydiscover_python')) {
    $skyPythonValue = $local['skydiscover_python']
}
if ([string]::IsNullOrWhiteSpace($skyPythonValue)) {
    $skyPythonValue = Join-Path $skyRoot '.venv/Scripts/python.exe'
}
$skyPython = Resolve-LocalPath $skyPythonValue $skyRoot
if (-not $skyPython -or -not (Test-Path -LiteralPath $skyPython -PathType Leaf)) {
    throw "SkyDiscover Python is unavailable: $skyPython. Prepare SkyDiscover independently; this launcher will not modify its environment."
}

$evaluatorValue = $EvaluatorLauncher
$profileKey = if ($Profile -eq 'research-l10-r0') { 'l10_r0_python' } else { 'dtr_r0_python' }
$profileEnvironment = if ($Profile -eq 'research-l10-r0') {
    'BLINDASSIST_L10_R0_PYTHON'
} else {
    'BLINDASSIST_DTR_R0_PYTHON'
}
if ([string]::IsNullOrWhiteSpace($evaluatorValue) -and $local.ContainsKey($profileKey)) {
    $evaluatorValue = $local[$profileKey]
}
if ([string]::IsNullOrWhiteSpace($evaluatorValue)) {
    $evaluatorValue = [Environment]::GetEnvironmentVariable($profileEnvironment)
}
if ([string]::IsNullOrWhiteSpace($evaluatorValue) -and $local.ContainsKey('research_python')) {
    $evaluatorValue = $local['research_python']
}
$evaluator = Resolve-LocalPath $evaluatorValue
if (-not $evaluator -or -not (Test-Path -LiteralPath $evaluator -PathType Leaf)) {
    throw "BlindAssist evaluator launcher is unavailable for $Profile. Configure $profileKey or pass -EvaluatorLauncher."
}

$manifestPath = Resolve-LocalPath $Manifest
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "SkyDiscover assist manifest is unavailable: $manifestPath"
}

$skyCommit = (& git -C $skyRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($skyCommit)) {
    throw "Cannot resolve the SkyDiscover commit at $skyRoot"
}
Write-Output "skydiscover_commit: $skyCommit"
Write-Output "skydiscover_python: $skyPython"
Write-Output "evaluator_launcher: $evaluator"
Write-Output "manifest: $manifestPath"

$forward = @('-m', 'skydiscover.assist', $Command, $manifestPath, '--evaluator-launcher', $evaluator)
if ($Command -eq 'run' -and $Arguments.Count -gt 0) {
    $forward += '--'
    $forward += $Arguments
}
& $skyPython @forward
if ($LASTEXITCODE -ne 0) {
    throw "SkyDiscover assist $Command failed with exit code $LASTEXITCODE"
}
