param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$QairtRoot = "E:\codex-tools\qairt\2.47.0.260601",
    [string]$CachedDlcPath = "/data/local/tmp/ba_qairt_htp_r0/dav2-metric/518x686/output/model-sm8650-cached.dlc",
    [int]$DurationSeconds = 20,
    [int]$StressSeconds = 5,
    [int]$DepthPeriodMs = 500,
    [int]$TtlMs = 750,
    [switch]$IncludeGeometry,
    [switch]$PipelineGeometry,
    [switch]$PhaseLockedCadence,
    [switch]$NativeFp16Decode,
    [switch]$NativeGeometry,
    [switch]$NativeDirectDepthBridge,
    [switch]$DirectRgbBridge,
    [string]$OutputRoot,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
if ($DurationSeconds -lt 12) { throw "DurationSeconds must be at least 12" }
if ($StressSeconds -lt 3 -or $StressSeconds -ge $DurationSeconds) { throw "StressSeconds is invalid" }

function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$LogPath, [switch]$AllowFailure) {
    $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & $FilePath @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $oldPreference }
    if ($LogPath) { $output | Set-Content -LiteralPath $LogPath -Encoding utf8 }
    if ($code -ne 0 -and -not $AllowFailure) { throw "$FilePath failed with exit code $code" }
    [pscustomobject]@{ Lines = @($output | ForEach-Object { "$_" }); ExitCode = $code }
}
function Parse-Report([string[]]$Lines) {
    $match = [regex]::Match(($Lines -join "`n"), "camerax_latest_only_r0_report=(\{.*\})")
    if (-not $match.Success) { throw "CameraX latest-only report is missing" }
    $match.Groups[1].Value | ConvertFrom-Json
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not ((Invoke-Native $AdbPath @("devices") $null).Lines -match "^$([regex]::Escape($DeviceSerial))\s+device$")) {
    throw "USB device $DeviceSerial is not online"
}
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\camerax-latest-only-r0-$timestamp"
}
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists: $artifactRoot" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null
$appApk = Join-Path $repoRoot "app\build\outputs\apk\debug\app-debug.apk"
$testApk = Join-Path $repoRoot "apps\canaries\hftf-device-canary\build\outputs\apk\debug\hftf-device-canary-debug.apk"
$appApkSha256AtInstall = $null
$testApkSha256AtInstall = $null
Push-Location $repoRoot
try {
    if (-not $SkipBuild) {
        $env:JAVA_HOME = "E:\codex-tools\projects\blindassist\toolchain\.jdk\jdk17.0.19_10"
        $env:GRADLE_USER_HOME = "E:\codex-tools\projects\blindassist\state\gradle"
        $env:ANDROID_HOME = "E:\codex-tools\projects\blindassist\toolchain\android-sdk"
        $env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
        (Invoke-Native ".\gradlew.bat" @(
            ":app:assembleDebug", ":hftf-device-canary:assembleDebug", "-PqairtRoot=$QairtRoot",
            "--no-daemon", "--console=plain", "--max-workers=2"
        ) (Join-Path $artifactRoot "gradle-build.txt")).Lines | Out-Null
    }
    $appApkSha256AtInstall = (Get-FileHash -LiteralPath $appApk -Algorithm SHA256).Hash
    $testApkSha256AtInstall = (Get-FileHash -LiteralPath $testApk -Algorithm SHA256).Hash
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "input", "keyevent", "KEYCODE_WAKEUP") $null).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "wm", "dismiss-keyguard") $null -AllowFailure).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $appApk) (Join-Path $artifactRoot "install-app.txt")).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $testApk) (Join-Path $artifactRoot "install-test.txt")).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "chmod", "644", $CachedDlcPath) $null).Lines | Out-Null
    $instrument = Invoke-Native $AdbPath @(
        "-s", $DeviceSerial, "shell", "am", "instrument", "-w", "-r",
        "-e", "class", "com.linnan.blindassist.hftf.CameraXLatestOnlyDepthDeviceTest#realYuvLatestOnlyCachedQnn",
        "-e", "cachedDlcPath", $CachedDlcPath, "-e", "durationSeconds", "$DurationSeconds",
        "-e", "stressSeconds", "$StressSeconds", "-e", "depthPeriodMs", "$DepthPeriodMs",
        "-e", "ttlMs", "$TtlMs",
        "-e", "includeGeometry", "$($IncludeGeometry.IsPresent.ToString().ToLowerInvariant())",
        "-e", "pipelineGeometry", "$($PipelineGeometry.IsPresent.ToString().ToLowerInvariant())",
        "-e", "phaseLockedCadence", "$($PhaseLockedCadence.IsPresent.ToString().ToLowerInvariant())",
        "-e", "nativeFp16Decode", "$($NativeFp16Decode.IsPresent.ToString().ToLowerInvariant())",
        "-e", "nativeGeometry", "$($NativeGeometry.IsPresent.ToString().ToLowerInvariant())",
        "-e", "nativeDirectDepthBridge", "$($NativeDirectDepthBridge.IsPresent.ToString().ToLowerInvariant())",
        "-e", "directRgbBridge", "$($DirectRgbBridge.IsPresent.ToString().ToLowerInvariant())",
        "com.linnan.blindassist.hftf.devicecanary/androidx.test.runner.AndroidJUnitRunner"
    ) (Join-Path $artifactRoot "instrument.txt") -AllowFailure
    $report = Parse-Report $instrument.Lines
    $gitHead = (& git -C $repoRoot rev-parse HEAD).Trim()
    $dlcHashLine = (Invoke-Native $AdbPath @(
        "-s", $DeviceSerial, "shell", "sha256sum", $CachedDlcPath
    ) (Join-Path $artifactRoot "cached-dlc-sha256.txt")).Lines | Select-Object -First 1
    if ($dlcHashLine -notmatch '^([0-9a-fA-F]{64})\s+') { throw "cached DLC SHA-256 is unavailable" }
    $cachedDlcSha256 = $Matches[1].ToUpperInvariant()
    [ordered]@{
        schema = "blindassist_camerax_latest_only_r0_bundle"; generated_at = (Get-Date).ToString("o")
        device_serial = $DeviceSerial; transport = "usb"; cached_dlc_path = $CachedDlcPath
        device_model = ((Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "getprop", "ro.product.model") $null).Lines -join "").Trim()
        device_soc = ((Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "getprop", "ro.soc.model") $null).Lines -join "").Trim()
        device_android_release = ((Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "getprop", "ro.build.version.release") $null).Lines -join "").Trim()
        cached_dlc_sha256 = $cachedDlcSha256
        git_head = $gitHead
        app_apk_sha256 = $appApkSha256AtInstall
        test_apk_sha256 = $testApkSha256AtInstall
        instrumentation_exit_code = $instrument.ExitCode; report = $report
    } | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $artifactRoot "result.json") -Encoding utf8
    "artifact_root=$artifactRoot"; "gate_pass=$($report.gate_pass)"
    $p50 = if ($report.include_geometry) { $report.full_depth_geometry_ms.p50 } else { $report.yuv_to_fp16_plus_qnn_ms.p50 }
    "full_pipeline_p50_ms=$p50"
    if ($instrument.ExitCode -ne 0) { throw "instrumentation failed with exit code $($instrument.ExitCode)" }
    if (-not $report.gate_pass) { throw "CameraX latest-only device gate failed" }
} finally { Pop-Location }
