param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [ValidateSet("Both", "Awake", "Dozing")]
    [string]$Mode = "Both",
    [int]$Repetitions = 100,
    [string]$OutputRoot,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
if ($Repetitions -lt 100) { throw "Repetitions must be at least 100" }

function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$LogPath) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $FilePath @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($LogPath) { $output | Set-Content -LiteralPath $LogPath -Encoding utf8 }
    if ($code -ne 0) { throw "$FilePath $($Arguments -join ' ') failed with exit code $code" }
    return @($output | ForEach-Object { "$_" })
}

function Set-ScreenState([string]$State) {
    if ($State -eq "awake") {
        Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "input", "keyevent", "KEYCODE_WAKEUP") $null | Out-Null
    } else {
        Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "input", "keyevent", "KEYCODE_SLEEP") $null | Out-Null
    }
    Start-Sleep -Seconds 3
}

function Parse-Report([string[]]$Lines) {
    $joined = $Lines -join "`n"
    $match = [regex]::Match($joined, "cpu_boundary_microbench_r0_report=(\{.*\})")
    if (-not $match.Success) { throw "microbenchmark report is missing" }
    return $match.Groups[1].Value | ConvertFrom-Json
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$corpus = Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-android-parity-r0"
if (-not (Test-Path -LiteralPath (Join-Path $corpus "manifest.json"))) { throw "frozen parity corpus is missing" }
$devices = Invoke-Native $AdbPath @("devices") $null
if (-not ($devices -match "^$([regex]::Escape($DeviceSerial))\s+device$")) { throw "USB device $DeviceSerial is not online" }

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\cpu-boundary-microbench-r0-$timestamp"
}
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists: $artifactRoot" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null

$appApk = Join-Path $repoRoot "app\build\outputs\apk\dualLoopShadow\app-dualLoopShadow.apk"
$testApk = Join-Path $repoRoot "apps\canaries\hftf-device-canary\build\outputs\apk\dualLoopShadow\hftf-device-canary-dualLoopShadow.apk"
Push-Location $repoRoot
try {
    if (-not $SkipBuild) {
        $env:JAVA_HOME = "E:\codex-tools\jdk-17"
        $env:GRADLE_USER_HOME = "E:\codex-tools\projects\blindassist\state\gradle"
        $env:ANDROID_HOME = "E:\codex-tools\projects\blindassist\toolchain\android-sdk"
        $env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
        Invoke-Native ".\gradlew.bat" @(
            ":app:assembleDualLoopShadow",
            ":hftf-device-canary:assembleDualLoopShadow",
            "--no-daemon",
            "--console=plain",
            "--max-workers=2"
        ) (Join-Path $artifactRoot "gradle-build.txt") | Out-Null
    }
    Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $appApk) (Join-Path $artifactRoot "install-app.txt") | Out-Null
    Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $testApk) (Join-Path $artifactRoot "install-test.txt") | Out-Null
    $deviceRoot = "/data/local/tmp/hftf_cpu_boundary_microbench_r0"
    Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "mkdir", "-p", "$deviceRoot/clean") $null | Out-Null
    foreach ($name in @("rgb_640x480_uint8.npy", "normalized_nchw_fp32_1x3x518x686.npy")) {
        Invoke-Native $AdbPath @("-s", $DeviceSerial, "push", (Join-Path $corpus "clean\$name"), "$deviceRoot/clean/$name") (Join-Path $artifactRoot "push-$name.txt") | Out-Null
    }

    $states = if ($Mode -eq "Both") { @("awake", "dozing") } else { @($Mode.ToLowerInvariant()) }
    $reports = [ordered]@{}
    foreach ($state in $states) {
        Set-ScreenState $state
        $log = Join-Path $artifactRoot "instrument-$state.txt"
        $lines = Invoke-Native $AdbPath @(
            "-s", $DeviceSerial, "shell", "am", "instrument", "-w", "-r",
            "-e", "class", "com.linnan.blindassist.hftf.Dav2PreprocessOptimizationDeviceTest#cpuBoundaryMicrobench",
            "-e", "corpusRoot", $deviceRoot,
            "-e", "repetitions", "$Repetitions",
            "-e", "outputName", "cpu-boundary-microbench-$state-r0.json",
            "com.linnan.blindassist.hftf.devicecanary/androidx.test.runner.AndroidJUnitRunner"
        ) $log
        $reports[$state] = Parse-Report $lines
        $reports[$state] | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $artifactRoot "$state.json") -Encoding utf8
    }
    [ordered]@{
        schema = "blindassist_cpu_boundary_microbench_r0_bundle"
        generated_at = (Get-Date).ToString("o")
        device_serial = $DeviceSerial
        transport = "usb"
        repetitions = $Repetitions
        reports = $reports
    } | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath (Join-Path $artifactRoot "result.json") -Encoding utf8
    Write-Output "artifact_root=$artifactRoot"
} finally {
    Pop-Location
}
