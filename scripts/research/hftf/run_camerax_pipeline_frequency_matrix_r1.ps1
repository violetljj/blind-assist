param(
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [int]$DurationSeconds = 120,
    [int]$StressSeconds = 5,
    [int]$TtlMs = 750,
    [string]$OutputRoot,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_camerax_pipeline_frequency_matrix_r0.ps1"
$parameters = @{
    DeviceSerial = $DeviceSerial
    DurationSeconds = $DurationSeconds
    StressSeconds = $StressSeconds
    TtlMs = $TtlMs
    PhaseLockedCadence = $true
}
if ($OutputRoot) { $parameters.OutputRoot = $OutputRoot }
if ($SkipBuild) { $parameters.SkipBuild = $true }
& $runner @parameters
if ($LASTEXITCODE -ne 0) { throw "phase-locked frequency matrix failed with exit code $LASTEXITCODE" }
