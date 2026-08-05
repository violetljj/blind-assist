param(
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$Fp16ParityResultPath = "artifacts.local/evidence/hftf/dav2-fp16-native-decode-parity-r3-20260805/result.json",
    [string]$OutputRoot,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_camerax_pipeline_sustained_10min_r1.ps1"
$parameters = @{
    DeviceSerial = $DeviceSerial
    NativeFp16Decode = $true
    Fp16ParityResultPath = $Fp16ParityResultPath
}
if ($OutputRoot) { $parameters.OutputRoot = $OutputRoot }
if ($SkipBuild) { $parameters.SkipBuild = $true }
& $runner @parameters
if ($LASTEXITCODE -ne 0) { throw "native-decode ten-minute sustained gate failed with exit code $LASTEXITCODE" }
