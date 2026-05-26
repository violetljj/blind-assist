param(
    [int]$ImageLimit = 100,
    [int]$PureWarmup = 10,
    [int]$PureRuns = 100,
    [int]$AppRunsPerImage = 1,
    [int]$DefaultRegressionSeconds = 90,
    [switch]$SkipDefaultRegression,
    [string]$AdbPath
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $repoRoot $Path)
}

function Resolve-Adb([string]$RequestedPath) {
    if ($RequestedPath) {
        if (-not (Test-Path -LiteralPath $RequestedPath)) {
            throw "ADB not found at $RequestedPath"
        }
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }
    $localAdb = Join-Path $repoRoot ".android-sdk\platform-tools\adb.exe"
    if (Test-Path -LiteralPath $localAdb) {
        return (Resolve-Path -LiteralPath $localAdb).Path
    }
    $command = Get-Command adb -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw "ADB not found. Pass -AdbPath or install platform-tools."
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$LogPath
    )
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $FilePath @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($LogPath) {
        $output | Tee-Object -FilePath $LogPath
    }
    if ($code -ne 0) {
        throw "$FilePath $($Arguments -join ' ') failed with exit code $code"
    }
    return $output
}

function Get-SingleDevice([string]$Adb) {
    $output = Invoke-Native $Adb @("devices") $null
    $devices = @(
        $output |
            Where-Object { $_ -match "^\S+\s+device$" } |
            ForEach-Object { ($_ -split "\s+")[0] }
    )
    if ($devices.Count -ne 1) {
        throw "Expected exactly one online device, found $($devices.Count). Raw adb devices output: $($output -join ' | ')"
    }
    return $devices[0]
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = Join-Path $repoRoot "test-artifacts.local-yolo26n-device-benchmark-$timestamp"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$python = Resolve-RepoPath ".venv-export312\Scripts\python.exe"
$model = Resolve-RepoPath ".downloads\detector-lab\exports\yolo26n_fp16_320.tflite"
$manifest = Resolve-RepoPath ".downloads\detector-lab\datasets\coco100\coco100_manifest.json"
$apk = Resolve-RepoPath "app\build\outputs\apk\debug\app-debug.apk"
$aapt = Resolve-RepoPath ".android-sdk\build-tools\35.0.0\aapt.exe"
$adb = Resolve-Adb $AdbPath
$device = Get-SingleDevice $adb

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found: $python"
}
if (-not (Test-Path -LiteralPath $model)) {
    throw "yolo26n TFLite candidate not found: $model"
}

Push-Location $repoRoot
try {
    $env:JAVA_HOME = (Resolve-Path ".\.jdk\jdk17.0.19_10").Path
    $env:PATH = "$env:JAVA_HOME\bin;$((Resolve-Path '.\.android-sdk\platform-tools').Path);$env:PATH"
    $env:GRADLE_USER_HOME = (Resolve-Path ".\.gradle-local").Path

    Invoke-Native $python @("scripts\inspect_tflite.py", "--allow-any-shape", ".downloads\detector-lab\exports\yolo26n_fp16_320.tflite") (Join-Path $artifactRoot "inspect-yolo26n.txt") | Out-Null
    if (-not (Test-Path -LiteralPath $manifest)) {
        Invoke-Native $python @("scripts\prepare_coco100.py", "--sample-count", "$ImageLimit") (Join-Path $artifactRoot "prepare-coco100.txt") | Out-Null
    } else {
        "manifest_exists=$manifest" | Tee-Object -FilePath (Join-Path $artifactRoot "prepare-coco100.txt") | Out-Null
    }

    Invoke-Native ".\gradlew.bat" @(
        ":app:assembleDebug",
        ":app:assembleDebugAndroidTest",
        "--no-daemon",
        "--console=plain"
    ) (Join-Path $artifactRoot "gradle-assemble.txt") | Out-Null

    if (-not (Test-Path -LiteralPath $apk)) {
        throw "Debug APK not found after build: $apk"
    }
    Invoke-Native $aapt @("list", $apk) (Join-Path $artifactRoot "main-apk-assets.txt") | Out-Null
    $mainAssets = Get-Content -Path (Join-Path $artifactRoot "main-apk-assets.txt")
    if ($mainAssets -notcontains "assets/yolo11n_fp16_320.tflite") {
        throw "Default yolo11n asset is missing from the main debug APK."
    }
    if ($mainAssets -contains "assets/yolo26n_fp16_320.tflite") {
        throw "yolo26n unexpectedly entered the main debug APK assets."
    }

    Invoke-Native ".\gradlew.bat" @(
        ":app:connectedDebugAndroidTest",
        "-Pandroid.testInstrumentationRunnerArguments.class=com.linnan.blindassist.benchmark.Yolo26nDeviceBenchmarkTest",
        "-Pandroid.testInstrumentationRunnerArguments.imageLimit=$ImageLimit",
        "-Pandroid.testInstrumentationRunnerArguments.pureWarmup=$PureWarmup",
        "-Pandroid.testInstrumentationRunnerArguments.pureRuns=$PureRuns",
        "-Pandroid.testInstrumentationRunnerArguments.appRunsPerImage=$AppRunsPerImage",
        "-Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true",
        "--no-daemon",
        "--console=plain"
    ) (Join-Path $artifactRoot "connected-yolo26n-benchmark.txt") | Out-Null

    Invoke-Native $adb @("-s", $device, "logcat", "-d", "-s", "Yolo26nBenchmark", "BlindAssistPerf") (Join-Path $artifactRoot "logcat-yolo26n.txt") | Out-Null
    $deviceArtifactRoot = "/sdcard/Android/data/com.linnan.blindassist/files/yolo26n-benchmark"
    Invoke-Native $adb @("-s", $device, "pull", $deviceArtifactRoot, (Join-Path $artifactRoot "device-yolo26n-benchmark")) (Join-Path $artifactRoot "adb-pull-yolo26n.txt") | Out-Null

    if (-not $SkipDefaultRegression) {
        powershell -ExecutionPolicy Bypass -File .\scripts\run_device_regression.ps1 -SampleSeconds $DefaultRegressionSeconds 2>&1 |
            Tee-Object -FilePath (Join-Path $artifactRoot "default-device-regression.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "Default model device regression failed with exit code $LASTEXITCODE"
        }
    }

    $summary = [ordered]@{
        status = "passed"
        timestamp = $timestamp
        device = $device
        artifactRoot = $artifactRoot
        model = $model
        coco100Manifest = $manifest
        defaultRegression = -not [bool]$SkipDefaultRegression
    }
    $summary | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $artifactRoot "summary.json") -Encoding utf8
    Write-Host "Yolo26n device benchmark artifacts: $artifactRoot"
} finally {
    Pop-Location
}
