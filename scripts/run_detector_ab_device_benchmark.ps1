param(
    [ValidateSet("Coco100", "BlindAssistEvalSet")]
    [string]$DatasetKind = "Coco100",
    [ValidateSet("DetectorAb", "DepthFusion", "DepthFusionSweep", "SanpoTraversabilityOracle", "SegmentationCandidate")]
    [string]$ComparisonMode = "DetectorAb",
    [string]$DatasetRoot,
    [int]$ImageLimit = 100,
    [int]$PureWarmup = 10,
    [int]$PureRuns = 100,
    [int]$AppRunsPerImage = 3,
    [double]$MatchIouThreshold = 0.5,
    [ValidateSet("current", "center_near_sensitive", "center_near_strict", "critical_sensitive", "side_near_sensitive")]
    [string]$RiskConfig = "current",
    [switch]$RiskSweep,
    [int]$DefaultRegressionSeconds = 90,
    [switch]$SkipDefaultRegression,
    [string]$SegmentationModelPath = "artifacts.local\downloads\traversability-lab\exports\mobilenetv3_lraspp_int8_256.tflite",
    [string]$SegmentationModelAssetName = "mobilenetv3_lraspp_int8_256.tflite",
    [string]$AdbPath,
    [string]$PythonPath = "E:\codex-tools\bin\blindassist-python.cmd",
    [string]$GradleUserHome = "E:\codex-tools\projects\blindassist\state\gradle"
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
$artifactRoot = Join-Path (Join-Path $repoRoot "artifacts.local\evidence\detector-ab-device-benchmark") $timestamp
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$python = Resolve-RepoPath $PythonPath
$yolo11n = Resolve-RepoPath "app\src\main\assets\yolo11n_fp16_320.tflite"
$yolo26n = Resolve-RepoPath "artifacts.local\downloads\detector-lab\exports\yolo26n_fp16_320.tflite"
$manifest = Resolve-RepoPath "artifacts.local\downloads\detector-lab\datasets\coco100\coco100_manifest.json"
$annotations = Resolve-RepoPath "artifacts.local\downloads\detector-lab\datasets\coco100\coco100_annotations.json"
$defaultBlindAssistEvalSet = "artifacts.local\evidence\datasets\blindassist-evalset-20260527-impl"
$requestedBlindAssistEvalSet = if ($DatasetRoot) { $DatasetRoot } else { $defaultBlindAssistEvalSet }
$blindAssistEvalSet = Resolve-RepoPath $requestedBlindAssistEvalSet
$segmentationModel = Resolve-RepoPath $SegmentationModelPath
$resolvedGradleUserHome = Resolve-RepoPath $GradleUserHome
$apk = Resolve-RepoPath "app\build\outputs\apk\debug\app-debug.apk"
$aapt = Resolve-RepoPath ".android-sdk\build-tools\35.0.0\aapt.exe"
$adb = Resolve-Adb $AdbPath
$device = Get-SingleDevice $adb

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found: $python"
}
if (-not (Test-Path -LiteralPath $resolvedGradleUserHome -PathType Container)) {
    throw "Gradle user home not found: $resolvedGradleUserHome"
}
if (-not (Test-Path -LiteralPath $yolo11n)) {
    throw "yolo11n TFLite asset not found: $yolo11n"
}
if ($ComparisonMode -eq "DetectorAb" -and -not (Test-Path -LiteralPath $yolo26n)) {
    throw "yolo26n TFLite candidate not found: $yolo26n"
}
if ($ComparisonMode -eq "SegmentationCandidate" -and -not (Test-Path -LiteralPath $segmentationModel)) {
    throw "Benchmark-only segmentation INT8 model not found: $segmentationModel"
}
if ($DatasetKind -eq "BlindAssistEvalSet") {
    if (-not (Test-Path -LiteralPath (Join-Path $blindAssistEvalSet "manifest.jsonl"))) {
        throw "BlindAssist evalset manifest not found: $(Join-Path $blindAssistEvalSet 'manifest.jsonl')"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $blindAssistEvalSet "images\test"))) {
        throw "BlindAssist evalset images/test directory not found: $(Join-Path $blindAssistEvalSet 'images\test')"
    }
}

Push-Location $repoRoot
try {
    $env:JAVA_HOME = (Resolve-Path ".\.jdk\jdk17.0.19_10").Path
    $env:PATH = "$env:JAVA_HOME\bin;$((Resolve-Path '.\.android-sdk\platform-tools').Path);$env:PATH"
    $env:GRADLE_USER_HOME = (Resolve-Path -LiteralPath $resolvedGradleUserHome).Path
    New-Item -ItemType Directory -Force -Path ".\.android-home", ".\.kotlin-home" | Out-Null
    $env:ANDROID_USER_HOME = (Resolve-Path ".\.android-home").Path
    $env:KOTLIN_HOME = (Resolve-Path ".\.kotlin-home").Path
    $env:GRADLE_OPTS = "-Dkotlin.compiler.execution.strategy=in-process"

    Invoke-Native $python @("scripts\inspect_tflite.py") (Join-Path $artifactRoot "inspect-yolo11n.txt") | Out-Null
    if ($ComparisonMode -eq "DetectorAb") {
        Invoke-Native $python @("scripts\inspect_tflite.py", "--allow-any-shape", "artifacts.local\downloads\detector-lab\exports\yolo26n_fp16_320.tflite") (Join-Path $artifactRoot "inspect-yolo26n.txt") | Out-Null
    }
    if ($DatasetKind -eq "Coco100") {
        if (-not (Test-Path -LiteralPath $manifest) -or -not (Test-Path -LiteralPath $annotations)) {
            Invoke-Native $python @("scripts\prepare_coco100.py", "--sample-count", "$ImageLimit") (Join-Path $artifactRoot "prepare-coco100.txt") | Out-Null
        } else {
            "manifest_exists=$manifest`nannotations_exists=$annotations" |
                Tee-Object -FilePath (Join-Path $artifactRoot "prepare-coco100.txt") | Out-Null
        }
    }

    Invoke-Native ".\gradlew.bat" @(
        ":app:assembleDebug",
        ":device-benchmark:assembleDebug",
        "-PblindAssistEvalSetDir=$blindAssistEvalSet",
        "-PsegmentationBenchmarkModelPath=$segmentationModel",
        "-PsegmentationBenchmarkModelAssetName=$SegmentationModelAssetName",
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
    if ($mainAssets -contains "assets/$SegmentationModelAssetName") {
        throw "Benchmark-only segmentation model unexpectedly entered the main debug APK assets."
    }

    $riskConfigs = if ($RiskSweep) {
        @("current", "center_near_sensitive", "center_near_strict", "critical_sensitive", "side_near_sensitive")
    } else {
        @($RiskConfig)
    }
    foreach ($currentRiskConfig in $riskConfigs) {
        Invoke-Native ".\gradlew.bat" @(
            ":device-benchmark:connectedDebugAndroidTest",
            "-PblindAssistEvalSetDir=$blindAssistEvalSet",
            "-PsegmentationBenchmarkModelPath=$segmentationModel",
            "-PsegmentationBenchmarkModelAssetName=$SegmentationModelAssetName",
            "-Pandroid.testInstrumentationRunnerArguments.class=com.linnan.blindassist.benchmark.DetectorAbDeviceBenchmarkTest",
            "-Pandroid.testInstrumentationRunnerArguments.datasetKind=$DatasetKind",
            "-Pandroid.testInstrumentationRunnerArguments.comparisonMode=$ComparisonMode",
            "-Pandroid.testInstrumentationRunnerArguments.segmentationModelAsset=$SegmentationModelAssetName",
            "-Pandroid.testInstrumentationRunnerArguments.riskConfig=$currentRiskConfig",
            "-Pandroid.testInstrumentationRunnerArguments.imageLimit=$ImageLimit",
            "-Pandroid.testInstrumentationRunnerArguments.pureWarmup=$PureWarmup",
            "-Pandroid.testInstrumentationRunnerArguments.pureRuns=$PureRuns",
            "-Pandroid.testInstrumentationRunnerArguments.appRunsPerImage=$AppRunsPerImage",
            "-Pandroid.testInstrumentationRunnerArguments.matchIouThreshold=$MatchIouThreshold",
            "-Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true",
            "--no-daemon",
            "--console=plain"
        ) (Join-Path $artifactRoot "connected-detector-ab-benchmark-$currentRiskConfig.txt") | Out-Null
    }

    Invoke-Native $adb @("-s", $device, "logcat", "-d", "-s", "DetectorAbBenchmark", "BlindAssistPerf") (Join-Path $artifactRoot "logcat-detector-ab.txt") | Out-Null
    $deviceArtifactRoot = "/sdcard/Android/data/com.linnan.blindassist/files/detector-ab-benchmark"
    Invoke-Native $adb @("-s", $device, "pull", $deviceArtifactRoot, (Join-Path $artifactRoot "device-detector-ab-benchmark")) (Join-Path $artifactRoot "adb-pull-detector-ab.txt") | Out-Null

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
        yolo11n = $yolo11n
        yolo26n = $yolo26n
        segmentationModel = if ($ComparisonMode -eq "SegmentationCandidate") { $segmentationModel } else { $null }
        datasetKind = $DatasetKind
        comparisonMode = $ComparisonMode
        blindAssistEvalSet = $blindAssistEvalSet
        coco100Manifest = $manifest
        coco100Annotations = $annotations
        matchIouThreshold = $MatchIouThreshold
        appRunsPerImage = $AppRunsPerImage
        riskConfigs = $riskConfigs
        defaultRegression = -not [bool]$SkipDefaultRegression
    }
    $summary | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $artifactRoot "summary.json") -Encoding utf8
    Write-Host "Detector A/B device benchmark artifacts: $artifactRoot"
} finally {
    Pop-Location
}
