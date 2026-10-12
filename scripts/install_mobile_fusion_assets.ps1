[CmdletBinding()]
param([string]$Serial = 'R5CX10M8Y8X')
$ErrorActionPreference = 'Stop'
$adbExecutable = (Get-Command adb -CommandType Application | Select-Object -First 1).Source
$repoRoot = Split-Path -Parent $PSScriptRoot
$bundle = Join-Path $repoRoot 'artifacts.local/work/mobile-fusion-integration-20261012'
$model = Join-Path $repoRoot 'artifacts.local/work/dav2s-mobile-20261012/dav2s_686x518_fp32.dlc'
$sources = @($model) + @('camera.json','atom_remap.bin','cnh_strong_m3.onnx','cnh_strong_calibration.bin' | ForEach-Object { Join-Path $bundle $_ })
foreach ($file in $sources) { if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing prepared artifact: $file" } }
$taskRemote = '/data/local/tmp/blindassist-mobile-fusion-install'
function Adb([string[]]$Arguments) {
    & $adbExecutable -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) { throw "ADB failed: $($Arguments -join ' ')" }
}
Adb @('shell','mkdir','-p',$taskRemote)
try {
    Adb @('shell','run-as','com.linnan.blindassist','mkdir','-p','no_backup/fusion-experiment')
    $receipt = @()
    foreach ($file in $sources) {
        $name = Split-Path -Leaf $file
        Adb @('push',$file,"$taskRemote/$name")
        Adb @('shell','run-as','com.linnan.blindassist','cp',"$taskRemote/$name","no_backup/fusion-experiment/$name")
        $actual = (& $adbExecutable -s $Serial shell run-as com.linnan.blindassist sha256sum "no_backup/fusion-experiment/$name").Split(' ')[0]
        if ($LASTEXITCODE -ne 0) { throw "Cannot hash $name" }
        $expected = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expected -ne $actual) { throw "Device hash mismatch: $name" }
        $receipt += @{ name=$name; sha256=$expected; bytes=(Get-Item -LiteralPath $file).Length }
    }
    $receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $bundle 'device-assets-receipt.json') -Encoding utf8
} finally {
    # Only this script's explicit staging filenames; no shared directories or recordings.
    foreach ($file in $sources) {
        & $adbExecutable -s $Serial shell rm "$taskRemote/$(Split-Path -Leaf $file)"
    }
    & $adbExecutable -s $Serial shell rmdir $taskRemote
}
