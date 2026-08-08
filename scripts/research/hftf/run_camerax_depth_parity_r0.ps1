param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$PythonPath = "E:\codex-tools\bin\blindassist-python.cmd",
    [string]$QairtRoot = "E:\codex-tools\qairt\2.47.0.260601",
    [string]$DeviceQairtRoot = "/data/local/tmp/ba_qairt_htp_r0",
    [string]$CachedDlcPath = "/data/local/tmp/ba_qairt_htp_r0/dav2-metric/518x686/output/model-sm8650-cached.dlc",
    [int]$WarmupFrames = 12,
    [ValidateSet("BACK", "FRONT")]
    [string]$LensFacing = "BACK",
    [string]$OutputRoot,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
if ($WarmupFrames -lt 1 -or $WarmupFrames -gt 120) { throw "WarmupFrames must be in 1..120" }

function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$LogPath, [switch]$AllowFailure) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & $FilePath @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $oldPreference }
    if ($LogPath) { $output | Set-Content -LiteralPath $LogPath -Encoding utf8 }
    if ($code -ne 0 -and -not $AllowFailure) {
        throw "$FilePath $($Arguments -join ' ') failed with exit code $code"
    }
    [pscustomobject]@{ Lines = @($output | ForEach-Object { "$_" }); ExitCode = $code }
}

function Parse-Report([string[]]$Lines) {
    $match = [regex]::Match(($Lines -join "`n"), "camerax_depth_parity_capture_r0_report=(\{.*\})")
    if (-not $match.Success) { throw "CameraX parity capture report is missing" }
    $match.Groups[1].Value | ConvertFrom-Json
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\camerax-depth-parity-r0-$timestamp"
}
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists: $artifactRoot" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null
if (-not ((Invoke-Native $AdbPath @("devices") $null).Lines -match "^$([regex]::Escape($DeviceSerial))\s+device$")) {
    throw "USB device $DeviceSerial is not online"
}

$appApk = Join-Path $repoRoot "app\build\outputs\apk\debug\app-debug.apk"
$testApk = Join-Path $repoRoot "apps\canaries\hftf-device-canary\build\outputs\apk\debug\hftf-device-canary-debug.apk"
$outputName = "camerax-depth-parity-r0-$timestamp"
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
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "input", "keyevent", "KEYCODE_WAKEUP") $null).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "wm", "dismiss-keyguard") $null -AllowFailure).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $appApk) (Join-Path $artifactRoot "install-app.txt")).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $testApk) (Join-Path $artifactRoot "install-test.txt")).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "chmod", "644", $CachedDlcPath) $null).Lines | Out-Null

    $instrument = Invoke-Native $AdbPath @(
        "-s", $DeviceSerial, "shell", "am", "instrument", "-w", "-r",
        "-e", "class", "com.linnan.blindassist.hftf.CameraXDepthParityDeviceTest#captureSameFrameParityEvidence",
        "-e", "cachedDlcPath", $CachedDlcPath,
        "-e", "warmupFrames", "$WarmupFrames",
        "-e", "lensFacing", $LensFacing,
        "-e", "outputName", $outputName,
        "com.linnan.blindassist.hftf.devicecanary/androidx.test.runner.AndroidJUnitRunner"
    ) (Join-Path $artifactRoot "instrument.txt") -AllowFailure
    $captureReport = Parse-Report $instrument.Lines
    if (-not $captureReport.gate_pass) { throw "device capture/input-buffer gate failed" }

    $captureRoot = Join-Path $artifactRoot "capture"
    New-Item -ItemType Directory -Path $captureRoot | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "pull", "$($captureReport.output_root)/.", $captureRoot) (Join-Path $artifactRoot "adb-pull.txt")).Lines | Out-Null
    (Invoke-Native $PythonPath @(
        "scripts/research/hftf/analyze_camerax_depth_parity_r0.py", "--capture-root", $captureRoot
    ) (Join-Path $artifactRoot "host-analysis-before-cli.txt")).Lines | Out-Null

    $deviceRunRoot = "$DeviceQairtRoot/camerax-depth-parity-r0-$timestamp"
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "mkdir", "-p", $deviceRunRoot) $null).Lines | Out-Null
    $deviceInput = "$deviceRunRoot/native-normalized-nchw-f16.raw"
    $deviceInputList = "$deviceRunRoot/input-list.txt"
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "push", (Join-Path $captureRoot "native_normalized_nchw_fp16.raw"), $deviceInput) (Join-Path $artifactRoot "adb-push-input.txt")).Lines | Out-Null
    $inputList = Join-Path $artifactRoot "input-list.txt"
    "image:=$deviceInput" | Set-Content -LiteralPath $inputList -Encoding ascii
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "push", $inputList, $deviceInputList) $null).Lines | Out-Null
    $cliCommand = "cd $DeviceQairtRoot && export LD_LIBRARY_PATH=$DeviceQairtRoot && export ADSP_LIBRARY_PATH=$DeviceQairtRoot\;/system/lib/rfsa/adsp\;/system/vendor/lib/rfsa/adsp\;/dsp && ./qnn-net-run --backend $DeviceQairtRoot/libQnnHtp.so --model $DeviceQairtRoot/libQnnModelDlc.so --dlc_path $CachedDlcPath --input_list $deviceInputList --output_dir $deviceRunRoot/output --use_native_input_files --use_native_output_files --perf_profile sustained_high_performance --log_level info"
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", $cliCommand) (Join-Path $artifactRoot "qnn-cli.txt")).Lines | Out-Null
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "pull", "$deviceRunRoot/output/Result_0/depth_m_native.raw", (Join-Path $captureRoot "cli_qnn_depth_fp16.raw")) $null).Lines | Out-Null

    (Invoke-Native $PythonPath @(
        "scripts/research/hftf/analyze_camerax_depth_parity_r0.py", "--capture-root", $captureRoot
    ) (Join-Path $artifactRoot "host-analysis-final.txt")).Lines | Out-Null
    $parity = Get-Content -LiteralPath (Join-Path $captureRoot "parity.json") -Raw | ConvertFrom-Json
    [ordered]@{
        schema = "blindassist_camerax_depth_parity_r0_bundle"
        generated_at = (Get-Date).ToString("o")
        device_serial = $DeviceSerial
        transport = "usb"
        instrumentation_exit_code = $instrument.ExitCode
        capture_report = $captureReport
        parity = $parity
    } | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath (Join-Path $artifactRoot "result.json") -Encoding utf8
    "artifact_root=$artifactRoot"
    "gate_pass=$($parity.gate_pass)"
    "gate_failures=$($parity.gate_failures -join ',')"
} finally {
    Pop-Location
}
