param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$QairtRoot = "E:\codex-tools\qairt\2.47.0.260601",
    [string]$CachedDlcPath = "/data/local/tmp/ba_qairt_htp_r0/dav2-metric/518x686/output/model-sm8650-cached.dlc",
    [string]$CliOutputPath = "/data/local/tmp/ba_qairt_htp_r0/dav2-metric/518x686/output-clean-native-f16-r0/Result_0/depth_m.raw",
    [string]$CorpusRoot = "/data/local/tmp/hftf_cpu_boundary_microbench_r0",
    [int]$Repetitions = 10,
    [string]$OutputRoot,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
if ($Repetitions -lt 10) { throw "Repetitions must be at least 10" }

function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$LogPath, [switch]$AllowFailure) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $FilePath @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($LogPath) { $output | Set-Content -LiteralPath $LogPath -Encoding utf8 }
    if ($code -ne 0 -and -not $AllowFailure) {
        throw "$FilePath $($Arguments -join ' ') failed with exit code $code"
    }
    return [pscustomobject]@{ Lines = @($output | ForEach-Object { "$_" }); ExitCode = $code }
}

function Parse-Report([string[]]$Lines) {
    $joined = $Lines -join "`n"
    $match = [regex]::Match($joined, "qnn_native_cached_context_r0_report=(\{.*\})")
    if (-not $match.Success) { throw "cached-context report is missing" }
    return $match.Groups[1].Value | ConvertFrom-Json
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not (Test-Path -LiteralPath (Join-Path $QairtRoot "lib\aarch64-android\libQnnHtp.so"))) {
    throw "QAIRT runtime is missing: $QairtRoot"
}
$devices = (Invoke-Native $AdbPath @("devices") $null).Lines
if (-not ($devices -match "^$([regex]::Escape($DeviceSerial))\s+device$")) {
    throw "USB device $DeviceSerial is not online"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\qnn-native-cached-context-r0-$timestamp"
}
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists: $artifactRoot" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null

$appApk = Join-Path $repoRoot "app\build\outputs\apk\debug\app-debug.apk"
$testApk = Join-Path $repoRoot "hftf-device-canary\build\outputs\apk\debug\hftf-device-canary-debug.apk"
Push-Location $repoRoot
try {
    if (-not $SkipBuild) {
        $env:JAVA_HOME = "E:\codex-tools\projects\blindassist\toolchain\.jdk\jdk17.0.19_10"
        $env:GRADLE_USER_HOME = "E:\codex-tools\projects\blindassist\state\gradle"
        $env:ANDROID_HOME = "E:\codex-tools\projects\blindassist\toolchain\android-sdk"
        $env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
        (Invoke-Native ".\gradlew.bat" @(
            ":app:assembleDebug",
            ":hftf-device-canary:assembleDebug",
            "-PqairtRoot=$QairtRoot",
            "--no-daemon",
            "--console=plain",
            "--max-workers=2"
        ) (Join-Path $artifactRoot "gradle-build.txt")).Lines | Out-Null
    }
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $appApk) (Join-Path $artifactRoot "install-app.txt")).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $testApk) (Join-Path $artifactRoot "install-test.txt")).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "chmod", "644", $CachedDlcPath, $CliOutputPath) $null).Lines | Out-Null

    $instrument = Invoke-Native $AdbPath @(
        "-s", $DeviceSerial, "shell", "am", "instrument", "-w", "-r",
        "-e", "class", "com.linnan.blindassist.hftf.Dav2QnnCachedContextDeviceTest#cachedContextExecuteAndFp16Parity",
        "-e", "cachedDlcPath", $CachedDlcPath,
        "-e", "corpusRoot", $CorpusRoot,
        "-e", "cliOutputPath", $CliOutputPath,
        "-e", "repetitions", "$Repetitions",
        "com.linnan.blindassist.hftf.devicecanary/androidx.test.runner.AndroidJUnitRunner"
    ) (Join-Path $artifactRoot "instrument.txt") -AllowFailure
    $report = Parse-Report $instrument.Lines
    $runtimeHash = (Get-FileHash -LiteralPath (Join-Path $QairtRoot "lib\aarch64-android\libQnnHtp.so") -Algorithm SHA256).Hash
    [ordered]@{
        schema = "blindassist_qnn_native_cached_context_r0_bundle"
        generated_at = (Get-Date).ToString("o")
        device_serial = $DeviceSerial
        transport = "usb"
        qairt_root = $QairtRoot
        qnn_htp_sha256 = $runtimeHash
        cached_dlc_path = $CachedDlcPath
        cli_output_path = $CliOutputPath
        instrumentation_exit_code = $instrument.ExitCode
        report = $report
    } | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath (Join-Path $artifactRoot "result.json") -Encoding utf8
    Write-Output "artifact_root=$artifactRoot"
    Write-Output "app_cli_gate_pass=$($report.app_cli_gate_pass)"
    Write-Output "fp16_preprocess_depth_gate_pass=$($report.fp16_preprocess_depth_gate_pass)"
} finally {
    Pop-Location
}
