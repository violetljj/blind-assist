[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$QairtRoot = 'E:\codex-tools\qairt\2.47.0.260601',
    [string]$MsvcRoot = 'E:\codex-tools\msvc-buildtools-2022'
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$hftfRoot = (Resolve-Path (Join-Path $scriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot '..\..\..\..\..')).Path
$artifactsRoot = (Resolve-Path (Join-Path $repoRoot 'artifacts.local')).Path
$outputParent = Split-Path -Parent $OutputRoot
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
if (-not (Resolve-Path $outputParent).Path.StartsWith($artifactsRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must be under $artifactsRoot"
}
if (Test-Path -LiteralPath $OutputRoot) { throw "OutputRoot already exists: $OutputRoot" }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null

$source = Join-Path $hftfRoot 'depthart_selective_scan_converter_op.cpp'
$vcvars = Join-Path $MsvcRoot 'VC\Auxiliary\Build\vcvars64.bat'
$include = Join-Path $QairtRoot 'include\QNN'
foreach ($path in ($source, $vcvars, $include)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
$dll = Join-Path $OutputRoot 'depthart_selective_scan_converter_op.dll'
$obj = Join-Path $OutputRoot 'depthart_selective_scan_converter_op.obj'
$command = 'call "' + $vcvars + '" >nul && cl.exe /nologo /std:c++17 /EHsc /LD /I"' +
    $include + '" /Fo:"' + $obj + '" /Fe:"' + $dll + '" "' + $source + '"'
cmd.exe /d /c $command
if ($LASTEXITCODE -ne 0) { throw "converter op package compile failed: $LASTEXITCODE" }

$receipt = [ordered]@{
    schema_version = 1
    status = 'COMPILED_CONVERTER_INFERENCE_ONLY'
    dll = [ordered]@{ path = $dll; bytes = (Get-Item $dll).Length; sha256 = (Get-FileHash $dll -Algorithm SHA256).Hash }
    authority = 'Converter shape/datatype inference only; no runtime kernel or HTP execution claim.'
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputRoot 'build-receipt.json') -Encoding utf8
$receipt
