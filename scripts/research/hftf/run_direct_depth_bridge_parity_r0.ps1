param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$QairtRoot = "E:\codex-tools\qairt\2.47.0.260601",
    [string]$HostFp16Path = "artifacts.local/evidence/hftf/camerax-depth-parity-r0-20260804-202405/capture/app_qnn_depth_fp16.raw",
    [int]$Repetitions = 100,
    [string]$OutputRoot,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
if ($Repetitions -lt 100) { throw "Repetitions must be at least 100" }
function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$LogPath, [switch]$AllowFailure) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & $FilePath @Arguments 2>&1; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
    if ($LogPath) { $output | Set-Content -LiteralPath $LogPath -Encoding utf8 }
    if ($code -ne 0 -and -not $AllowFailure) { throw "$FilePath failed with exit code $code" }
    [pscustomobject]@{ Lines = @($output | ForEach-Object { "$_" }); ExitCode = $code }
}
function Parse-Report([string[]]$Lines) {
    $match = [regex]::Match(($Lines -join "`n"), "dav2_direct_depth_bridge_parity_r0_report=(\{.*\})")
    if (-not $match.Success) { throw "direct depth bridge report is missing" }
    $match.Groups[1].Value | ConvertFrom-Json
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$hostInput = (Resolve-Path -LiteralPath (Join-Path $repoRoot $HostFp16Path)).Path
if ((Get-Item -LiteralPath $hostInput).Length -ne 710696) { throw "expected a 710696-byte FP16 depth map" }
if (-not ((Invoke-Native $AdbPath @("devices") $null).Lines -match "^$([regex]::Escape($DeviceSerial))\s+device$")) { throw "USB device is offline" }
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else { Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-direct-depth-bridge-parity-r0-$timestamp" }
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null
$deviceInput = "/data/local/tmp/ba-direct-depth-bridge-r0-$timestamp.raw"
$testApk = Join-Path $repoRoot "hftf-device-canary\build\outputs\apk\debug\hftf-device-canary-debug.apk"
Push-Location $repoRoot
try {
    if (-not $SkipBuild) {
        $env:JAVA_HOME = "E:\codex-tools\projects\blindassist\toolchain\.jdk\jdk17.0.19_10"
        $env:GRADLE_USER_HOME = "E:\codex-tools\projects\blindassist\state\gradle"
        $env:ANDROID_HOME = "E:\codex-tools\projects\blindassist\toolchain\android-sdk"; $env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
        (Invoke-Native ".\gradlew.bat" @(
            ":hftf-device-canary:assembleDebug", "-PqairtRoot=$QairtRoot",
            "--no-daemon", "--console=plain", "--max-workers=2"
        ) (Join-Path $artifactRoot "gradle-build.txt")).Lines | Out-Null
    }
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "push", $hostInput, $deviceInput) (Join-Path $artifactRoot "push-input.txt")).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $testApk) (Join-Path $artifactRoot "install-test.txt")).Lines | Out-Null
    $instrument = Invoke-Native $AdbPath @(
        "-s", $DeviceSerial, "shell", "am", "instrument", "-w", "-r",
        "-e", "class", "com.linnan.blindassist.hftf.Dav2DirectDepthBridgeParityDeviceTest#fusedDecodeResizeAndDirectGeometryMatchStagedPath",
        "-e", "depthFp16Path", $deviceInput, "-e", "repetitions", "$Repetitions",
        "com.linnan.blindassist.hftf.devicecanary/androidx.test.runner.AndroidJUnitRunner"
    ) (Join-Path $artifactRoot "instrument.txt") -AllowFailure
    $report = Parse-Report $instrument.Lines
    [ordered]@{
        schema = "blindassist_dav2_direct_depth_bridge_parity_r0_bundle"; generated_at = (Get-Date).ToString("o")
        device_serial = $DeviceSerial; host_input_sha256 = (Get-FileHash -LiteralPath $hostInput -Algorithm SHA256).Hash
        test_apk_sha256 = (Get-FileHash -LiteralPath $testApk -Algorithm SHA256).Hash
        instrumentation_exit_code = $instrument.ExitCode; report = $report
    } | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $artifactRoot "result.json") -Encoding utf8
    "artifact_root=$artifactRoot"; "gate_pass=$($report.gate_pass)"
    if ($instrument.ExitCode -ne 0 -or -not $report.gate_pass) { throw "direct depth bridge parity gate failed" }
} finally { Pop-Location }
