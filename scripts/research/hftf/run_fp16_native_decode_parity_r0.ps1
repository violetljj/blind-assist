param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [ValidateSet("fp16_decode", "direct_rgb_bridge")]
    [string]$ParityKind = "fp16_decode",
    [string]$OutputRoot,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$LogPath, [switch]$AllowFailure) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & $FilePath @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($LogPath) { $output | Set-Content -LiteralPath $LogPath -Encoding utf8 }
    if ($code -ne 0 -and -not $AllowFailure) { throw "$FilePath failed with exit code $code" }
    [pscustomobject]@{ Lines = @($output | ForEach-Object { "$_" }); ExitCode = $code }
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else {
    Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-fp16-native-decode-parity-r0-$timestamp"
}
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists: $artifactRoot" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null
$testApk = Join-Path $repoRoot "hftf-device-canary\build\outputs\apk\debug\hftf-device-canary-debug.apk"
$testClass = if ($ParityKind -eq "direct_rgb_bridge") {
    "com.linnan.blindassist.hftf.Dav2DirectRgbBridgeParityDeviceTest#directRgbAndCanonicalTensorAreBitExactForEveryRotation"
} else {
    "com.linnan.blindassist.hftf.Dav2Fp16DecodeParityDeviceTest#everyHalfBitPatternMatchesAndroidHalf"
}
$reportKey = if ($ParityKind -eq "direct_rgb_bridge") {
    "dav2_direct_rgb_bridge_parity_r0_report"
} else {
    "dav2_fp16_native_decode_parity_r0_report"
}
Push-Location $repoRoot
try {
    if (-not $SkipBuild) {
        $env:JAVA_HOME = "E:\codex-tools\jdk-17"
        (Invoke-Native ".\gradlew.bat" @(
            ":hftf-device-canary:assembleDebug", "--no-daemon", "--console=plain", "--max-workers=2"
        ) (Join-Path $artifactRoot "gradle-build.txt")).Lines | Out-Null
    }
    (Invoke-Native $AdbPath @("-s", $DeviceSerial, "install", "-r", $testApk) (
        Join-Path $artifactRoot "install-test.txt"
    )).Lines | Out-Null
    $instrument = Invoke-Native $AdbPath @(
        "-s", $DeviceSerial, "shell", "am", "instrument", "-w", "-r",
        "-e", "class", $testClass,
        "com.linnan.blindassist.hftf.devicecanary/androidx.test.runner.AndroidJUnitRunner"
    ) (Join-Path $artifactRoot "instrument.txt") -AllowFailure
    $match = [regex]::Match(($instrument.Lines -join "`n"), "$reportKey=(\{.*\})")
    if (-not $match.Success) { throw "$ParityKind parity report is missing" }
    $report = $match.Groups[1].Value | ConvertFrom-Json
    $resultPath = Join-Path $artifactRoot "result.json"
    [ordered]@{
        schema = "blindassist_dav2_${ParityKind}_parity_r0_bundle"
        generated_at = (Get-Date).ToString("o")
        device_serial = $DeviceSerial
        instrumentation_exit_code = $instrument.ExitCode
        test_apk_sha256 = (Get-FileHash -LiteralPath $testApk -Algorithm SHA256).Hash
        native_source_sha256 = (Get-FileHash -LiteralPath (
            Join-Path $repoRoot "hftf-device-canary/src/main/cpp/dav2_preprocess_native.cpp"
        ) -Algorithm SHA256).Hash
        yuv_source_sha256 = (Get-FileHash -LiteralPath (
            Join-Path $repoRoot "hftf-device-canary/src/main/cpp/dav2_yuv420_rgb.cpp"
        ) -Algorithm SHA256).Hash
        report = $report
    } | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $resultPath -Encoding utf8
    "artifact_root=$artifactRoot"
    "result_sha256=$((Get-FileHash -LiteralPath $resultPath -Algorithm SHA256).Hash)"
    "gate_pass=$($report.pass)"
    if ($instrument.ExitCode -ne 0 -or -not $report.pass) { throw "$ParityKind parity failed" }
} finally { Pop-Location }
