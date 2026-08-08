param(
    [string]$AdbPath,
    [string]$DeviceSerial,
    [int]$SustainedSeconds = 600,
    [string]$OutputRoot,
    [switch]$SkipBuild,
    [switch]$SkipShortArms
)

$ErrorActionPreference = "Stop"

function Invoke-Native {
    param([string]$FilePath, [string[]]$Arguments, [string]$LogPath)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $FilePath @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($LogPath) { $output | Tee-Object -FilePath $LogPath | Out-Null }
    if ($code -ne 0) {
        throw "$FilePath $($Arguments -join ' ') failed with exit code $code"
    }
    return @($output | ForEach-Object { "$_" })
}

function Resolve-Adb([string]$Requested) {
    if ($Requested) { return (Resolve-Path -LiteralPath $Requested).Path }
    $command = Get-Command adb -ErrorAction SilentlyContinue
    if (-not $command) { throw "adb is unavailable; pass -AdbPath" }
    return $command.Source
}

function Get-OnlineDevice([string]$Adb, [string]$RequestedSerial) {
    $rows = Invoke-Native $Adb @("devices") $null
    $devices = @(
        $rows | Where-Object { $_ -match "^(\S+)\s+device$" } |
            ForEach-Object { ($_ -split "\s+")[0] }
    )
    if ($RequestedSerial) {
        if ($devices -notcontains $RequestedSerial) { throw "requested device $RequestedSerial is not online" }
        return $RequestedSerial
    }
    if ($devices.Count -ne 1) { throw "expected one online device; found $($devices.Count)" }
    return $devices[0]
}

function Assert-Hash([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "missing file $Path" }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -ne $Expected) { throw "hash mismatch for $Path expected=$Expected actual=$actual" }
}

function Parse-Report([string[]]$Lines, [string]$Key) {
    $joined = $Lines -join "`n"
    $match = [regex]::Match($joined, [regex]::Escape($Key) + "=(\{.*\})")
    if (-not $match.Success) { throw "instrumentation report $Key is missing" }
    return ($match.Groups[1].Value | ConvertFrom-Json)
}

function Run-Instrumentation {
    param(
        [string]$Adb,
        [string]$Serial,
        [string]$ClassMethod,
        [hashtable]$Arguments,
        [string]$LogPath
    )
    $command = @("-s", $Serial, "shell", "am", "instrument", "-w", "-r", "-e", "class", $ClassMethod)
    foreach ($key in ($Arguments.Keys | Sort-Object)) {
        $command += @("-e", $key, "$($Arguments[$key])")
    }
    $command += "com.linnan.blindassist.hftf.devicecanary/androidx.test.runner.AndroidJUnitRunner"
    return Invoke-Native $Adb $command $LogPath
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$adb = Resolve-Adb $AdbPath
$serial = Get-OnlineDevice $adb $DeviceSerial
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) {
    [System.IO.Path]::GetFullPath($OutputRoot)
} else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-android-performance-r0-$timestamp"
}
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists: $artifactRoot" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null

$onnx = Join-Path $repoRoot "artifacts.local\models\dav2-metric-hypersim-vits-android-r0\model_518x686.onnx"
$tflite = Join-Path $repoRoot "artifacts.local\models\dav2-metric-hypersim-vits-android-r0\model_518x686_fp32.tflite"
$corpus = Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-android-parity-r0"
$appApk = Join-Path $repoRoot "app\build\outputs\apk\dualLoopShadow\app-dualLoopShadow.apk"
$testApk = Join-Path $repoRoot "apps\canaries\hftf-device-canary\build\outputs\apk\dualLoopShadow\hftf-device-canary-dualLoopShadow.apk"
Assert-Hash $onnx "870339770E21675830F7E2020983DDA058752D237C8B86951ED1E6F9A6243D01"
Assert-Hash $tflite "0277FBC74C73D95433B43BEE9D61DD08F1E79B67A2F64A6DA871F3A23FBED8E3"
if (-not (Test-Path -LiteralPath (Join-Path $corpus "manifest.json"))) { throw "parity corpus is missing" }

Push-Location $repoRoot
try {
    if (-not $SkipBuild) {
        $env:JAVA_HOME = "E:\codex-tools\jdk-17"
        $env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
        Invoke-Native ".\gradlew.bat" @(
            ":app:assembleDualLoopShadow",
            ":hftf-device-canary:assembleDualLoopShadow",
            "--no-daemon",
            "--console=plain"
        ) (Join-Path $artifactRoot "gradle-build.txt") | Out-Null
    }
    Invoke-Native $adb @("-s", $serial, "install", "-r", $appApk) (Join-Path $artifactRoot "install-app.txt") | Out-Null
    Invoke-Native $adb @("-s", $serial, "install", "-r", $testApk) (Join-Path $artifactRoot "install-test.txt") | Out-Null
    $deviceRoot = "/data/local/tmp/hftf_dav2_metric_android_r0"
    Invoke-Native $adb @("-s", $serial, "shell", "mkdir", "-p", "$deviceRoot/corpus") $null | Out-Null
    Invoke-Native $adb @("-s", $serial, "push", $onnx, "$deviceRoot/model.onnx") (Join-Path $artifactRoot "push-onnx.txt") | Out-Null
    Invoke-Native $adb @("-s", $serial, "push", $tflite, "$deviceRoot/model_fp32.tflite") (Join-Path $artifactRoot "push-tflite.txt") | Out-Null
    Invoke-Native $adb @("-s", $serial, "push", "$corpus\.", "$deviceRoot/corpus/") (Join-Path $artifactRoot "push-corpus.txt") | Out-Null

    $reports = [ordered]@{}
    if (-not $SkipShortArms) {
        $lines = Run-Instrumentation $adb $serial `
            "com.linnan.blindassist.hftf.Dav2MetricAndroidParityTest#frozenStressCorpusMatchesPytorchAndPipelineReferences" `
            @{ modelPath = "$deviceRoot/model.onnx"; corpusRoot = "$deviceRoot/corpus" } `
            (Join-Path $artifactRoot "instrument-parity.txt")
        $reports.parity = Parse-Report $lines "dav2_metric_android_parity_report"

        $lines = Run-Instrumentation $adb $serial `
            "com.linnan.blindassist.hftf.Dav2MetricAndroidPerformanceTest#fixedGradientParityAndShortPerformance" `
            @{ modelPath = "$deviceRoot/model.onnx"; backend = "cpu" } `
            (Join-Path $artifactRoot "instrument-onnx-short.txt")
        $reports.onnx_short = Parse-Report $lines "dav2_metric_android_short_performance_report"

        $lines = Run-Instrumentation $adb $serial `
            "com.linnan.blindassist.hftf.Dav2MetricAndroidTfliteTest#fixedGradientParityAndShortPerformance" `
            @{ modelPath = "$deviceRoot/model_fp32.tflite" } `
            (Join-Path $artifactRoot "instrument-tflite-short.txt")
        $reports.tflite_short = Parse-Report $lines "dav2_metric_android_tflite_short_performance_report"
    }

    $thermalPath = Join-Path $artifactRoot "host-thermal-memory-5s.jsonl"
    $deadline = (Get-Date).AddSeconds($SustainedSeconds + 180)
    $thermalJob = Start-Job -ScriptBlock {
        param($Adb, $Serial, $Deadline, $Path)
        while ((Get-Date) -lt $Deadline) {
            $thermal = (& $Adb -s $Serial shell dumpsys thermalservice 2>&1) -join "`n"
            $battery = (& $Adb -s $Serial shell dumpsys battery 2>&1) -join "`n"
            $meminfo = (& $Adb -s $Serial shell dumpsys meminfo com.linnan.blindassist.dualloop.shadow 2>&1) -join "`n"
            [ordered]@{
                timestamp = (Get-Date).ToString("o")
                thermalservice = $thermal
                battery = $battery
                meminfo = $meminfo
            } | ConvertTo-Json -Compress | Add-Content -LiteralPath $Path -Encoding utf8
            Start-Sleep -Seconds 5
        }
    } -ArgumentList $adb, $serial, $deadline, $thermalPath
    try {
        $lines = Run-Instrumentation $adb $serial `
            "com.linnan.blindassist.hftf.Dav2MetricAndroidSustainedTest#sustainedFullPipelinePerformance" `
            @{ modelPath = "$deviceRoot/model.onnx"; corpusRoot = "$deviceRoot/corpus"; durationMs = $SustainedSeconds * 1000 } `
            (Join-Path $artifactRoot "instrument-sustained.txt")
        $reports.sustained = Parse-Report $lines "dav2_metric_android_sustained_pipeline_report"
    } finally {
        Stop-Job $thermalJob -ErrorAction SilentlyContinue
        Receive-Job $thermalJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $thermalJob -Force -ErrorAction SilentlyContinue
    }

    $deviceProperties = Invoke-Native $adb @("-s", $serial, "shell", "getprop") (Join-Path $artifactRoot "device-getprop.txt")
    [ordered]@{
        schema = "blindassist_dav2_android_performance_r0_raw_bundle_v1"
        generated_at = (Get-Date).ToString("o")
        device_serial = $serial
        sustained_seconds = $SustainedSeconds
        onnx_sha256 = (Get-FileHash -LiteralPath $onnx -Algorithm SHA256).Hash
        tflite_sha256 = (Get-FileHash -LiteralPath $tflite -Algorithm SHA256).Hash
        reports = $reports
        device_properties_line_count = $deviceProperties.Count
    } | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $artifactRoot "result.json") -Encoding utf8
    Write-Output "artifact_root=$artifactRoot"
} finally {
    Pop-Location
}
