param(
    [string]$DatasetRoot = "test-artifacts.local\datasets\blindassist-evalset-20260527-impl",
    [string]$DepthModelPath = ".downloads\depth-lab\exports\depth_anything_v2_small_fp32.tflite",
    [string]$DepthModelAsset = "",
    [ValidateSet("DepthFusion", "DepthFusionSweep")]
    [string]$ComparisonMode = "DepthFusion",
    [string]$DepthCloserIsLarger = "true",
    [double]$DepthSamplePercentile = 0.50,
    [double]$DepthInnerCropRatio = 1.0,
    [string]$DepthLowerHalfOnly = "true",
    [int]$DepthMinSamples = 4,
    [double]$DepthMinLocalRange = 0.0,
    [double]$DepthMinConfidence = 0.55,
    [double]$DepthCriticalThreshold = 0.78,
    [double]$DepthNearThreshold = 0.58,
    [double]$DepthMidThreshold = 0.35,
    [int]$ImageLimit = 100,
    [int]$PureWarmup = 10,
    [int]$PureRuns = 100,
    [int]$AppRunsPerImage = 3,
    [double]$MatchIouThreshold = 0.5,
    [ValidateSet("current", "center_near_sensitive", "center_near_strict", "critical_sensitive", "side_near_sensitive")]
    [string]$RiskConfig = "current",
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

function Convert-ToBoolString([string]$Value) {
    $normalized = $Value.Trim().ToLowerInvariant()
    if ($normalized -in @("true", "1", "yes", "y")) {
        return "true"
    }
    if ($normalized -in @("false", "0", "no", "n")) {
        return "false"
    }
    throw "Expected boolean value, got: $Value"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = Join-Path (Join-Path $repoRoot "test-artifacts.local\depth-fusion-benchmark") $timestamp
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$python = Resolve-RepoPath ".venv-export312\Scripts\python.exe"
$yolo11n = Resolve-RepoPath "app\src\main\assets\yolo11n_fp16_320.tflite"
$depthModel = Resolve-RepoPath $DepthModelPath
$blindAssistEvalSet = Resolve-RepoPath $DatasetRoot
$apk = Resolve-RepoPath "app\build\outputs\apk\debug\app-debug.apk"
$aapt = Resolve-RepoPath ".android-sdk\build-tools\35.0.0\aapt.exe"
$adb = Resolve-Adb $AdbPath
$device = Get-SingleDevice $adb

if ([string]::IsNullOrWhiteSpace($DepthModelAsset)) {
    $DepthModelAsset = "depth/$([System.IO.Path]::GetFileName($depthModel))"
}
$depthModelAssetName = [System.IO.Path]::GetFileName($DepthModelAsset)
$depthCloserIsLargerValue = Convert-ToBoolString $DepthCloserIsLarger
$depthLowerHalfOnlyValue = Convert-ToBoolString $DepthLowerHalfOnly

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found: $python"
}
if (-not (Test-Path -LiteralPath $yolo11n)) {
    throw "yolo11n TFLite asset not found: $yolo11n"
}
if (-not (Test-Path -LiteralPath $depthModel)) {
    throw "Depth TFLite candidate not found: $depthModel"
}
if (-not (Test-Path -LiteralPath (Join-Path $blindAssistEvalSet "manifest.jsonl"))) {
    throw "BlindAssist evalset manifest not found: $(Join-Path $blindAssistEvalSet 'manifest.jsonl')"
}
if (-not (Test-Path -LiteralPath (Join-Path $blindAssistEvalSet "images\test"))) {
    throw "BlindAssist evalset images/test directory not found: $(Join-Path $blindAssistEvalSet 'images\test')"
}

Push-Location $repoRoot
try {
    $env:JAVA_HOME = (Resolve-Path ".\.jdk\jdk17.0.19_10").Path
    $env:PATH = "$env:JAVA_HOME\bin;$((Resolve-Path '.\.android-sdk\platform-tools').Path);$env:PATH"
    $env:GRADLE_USER_HOME = (Resolve-Path ".\.gradle-local").Path
    New-Item -ItemType Directory -Force -Path ".\.android-home", ".\.kotlin-home" | Out-Null
    $env:ANDROID_USER_HOME = (Resolve-Path ".\.android-home").Path
    $env:KOTLIN_HOME = (Resolve-Path ".\.kotlin-home").Path
    $env:GRADLE_OPTS = "-Dkotlin.compiler.execution.strategy=in-process"

    Invoke-Native $python @("scripts\inspect_tflite.py") (Join-Path $artifactRoot "inspect-yolo11n.txt") | Out-Null
    Invoke-Native $python @("scripts\inspect_depth_model.py", $DepthModelPath, "--json-output", (Join-Path $artifactRoot "inspect-depth-model.json")) (Join-Path $artifactRoot "inspect-depth-model.txt") | Out-Null
    Invoke-Native $python @("scripts\smoke_depth_model.py", "--model", $DepthModelPath, "--dataset-root", $DatasetRoot, "--image-limit", "20", "--json-output", (Join-Path $artifactRoot "smoke-depth-model.json")) (Join-Path $artifactRoot "smoke-depth-model.txt") | Out-Null

    Invoke-Native ".\gradlew.bat" @(
        ":app:assembleDebug",
        ":app:assembleDebugAndroidTest",
        "-PblindAssistEvalSetDir=$blindAssistEvalSet",
        "-PdepthBenchmarkModelPath=$depthModel",
        "-PdepthBenchmarkModelAssetName=$depthModelAssetName",
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
    if ($mainAssets -contains "assets/depth/depth_anything_v2_small_fp32.tflite") {
        throw "Depth model unexpectedly entered the main debug APK assets."
    }
    if ($mainAssets -contains "assets/$DepthModelAsset") {
        throw "Depth model unexpectedly entered the main debug APK assets: assets/$DepthModelAsset"
    }

    Invoke-Native ".\gradlew.bat" @(
        ":app:connectedDebugAndroidTest",
        "-PblindAssistEvalSetDir=$blindAssistEvalSet",
        "-PdepthBenchmarkModelPath=$depthModel",
        "-PdepthBenchmarkModelAssetName=$depthModelAssetName",
        "-Pandroid.testInstrumentationRunnerArguments.class=com.linnan.blindassist.benchmark.DetectorAbDeviceBenchmarkTest",
        "-Pandroid.testInstrumentationRunnerArguments.datasetKind=BlindAssistEvalSet",
        "-Pandroid.testInstrumentationRunnerArguments.comparisonMode=$ComparisonMode",
        "-Pandroid.testInstrumentationRunnerArguments.depthModelAsset=$DepthModelAsset",
        "-Pandroid.testInstrumentationRunnerArguments.depthCloserIsLarger=$depthCloserIsLargerValue",
        "-Pandroid.testInstrumentationRunnerArguments.depthSamplePercentile=$DepthSamplePercentile",
        "-Pandroid.testInstrumentationRunnerArguments.depthInnerCropRatio=$DepthInnerCropRatio",
        "-Pandroid.testInstrumentationRunnerArguments.depthLowerHalfOnly=$depthLowerHalfOnlyValue",
        "-Pandroid.testInstrumentationRunnerArguments.depthMinSamples=$DepthMinSamples",
        "-Pandroid.testInstrumentationRunnerArguments.depthMinLocalRange=$DepthMinLocalRange",
        "-Pandroid.testInstrumentationRunnerArguments.depthMinConfidence=$DepthMinConfidence",
        "-Pandroid.testInstrumentationRunnerArguments.depthCriticalThreshold=$DepthCriticalThreshold",
        "-Pandroid.testInstrumentationRunnerArguments.depthNearThreshold=$DepthNearThreshold",
        "-Pandroid.testInstrumentationRunnerArguments.depthMidThreshold=$DepthMidThreshold",
        "-Pandroid.testInstrumentationRunnerArguments.riskConfig=$RiskConfig",
        "-Pandroid.testInstrumentationRunnerArguments.imageLimit=$ImageLimit",
        "-Pandroid.testInstrumentationRunnerArguments.pureWarmup=$PureWarmup",
        "-Pandroid.testInstrumentationRunnerArguments.pureRuns=$PureRuns",
        "-Pandroid.testInstrumentationRunnerArguments.appRunsPerImage=$AppRunsPerImage",
        "-Pandroid.testInstrumentationRunnerArguments.matchIouThreshold=$MatchIouThreshold",
        "-Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true",
        "--no-daemon",
        "--console=plain"
    ) (Join-Path $artifactRoot "connected-depth-fusion-benchmark.txt") | Out-Null

    Invoke-Native $adb @("-s", $device, "logcat", "-d", "-s", "DetectorAbBenchmark", "BlindAssistPerf", "TfliteDepthEstimator") (Join-Path $artifactRoot "logcat-depth-fusion.txt") | Out-Null
    $deviceArtifactRoot = "/sdcard/Android/data/com.linnan.blindassist/files/detector-ab-benchmark"
    Invoke-Native $adb @("-s", $device, "pull", $deviceArtifactRoot, (Join-Path $artifactRoot "device-depth-fusion-benchmark")) (Join-Path $artifactRoot "adb-pull-depth-fusion.txt") | Out-Null

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
        depthModel = $depthModel
        depthModelAsset = $DepthModelAsset
        comparisonMode = $ComparisonMode
        depthCloserIsLarger = $depthCloserIsLargerValue
        depthSampling = [ordered]@{
            samplePercentile = $DepthSamplePercentile
            innerCropRatio = $DepthInnerCropRatio
            lowerHalfOnly = $depthLowerHalfOnlyValue
            minSamples = $DepthMinSamples
            minLocalRange = $DepthMinLocalRange
            minConfidence = $DepthMinConfidence
            criticalThreshold = $DepthCriticalThreshold
            nearThreshold = $DepthNearThreshold
            midThreshold = $DepthMidThreshold
        }
        datasetKind = "BlindAssistEvalSet"
        blindAssistEvalSet = $blindAssistEvalSet
        matchIouThreshold = $MatchIouThreshold
        appRunsPerImage = $AppRunsPerImage
        riskConfig = $RiskConfig
        defaultRegression = -not [bool]$SkipDefaultRegression
    }
    $summary | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $artifactRoot "summary.json") -Encoding utf8
    Write-Host "Depth fusion benchmark artifacts: $artifactRoot"
} finally {
    Pop-Location
}
