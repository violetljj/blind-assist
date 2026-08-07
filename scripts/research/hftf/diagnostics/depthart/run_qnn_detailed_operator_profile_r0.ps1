param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$QairtRoot = "E:\codex-tools\qairt\2.47.0.260601",
    [int]$DurationSeconds = 10,
    [string]$OutputRoot
)
$ErrorActionPreference = "Stop"
if ($DurationSeconds -lt 1 -or $DurationSeconds -gt 60) { throw "DurationSeconds must be in 1..60" }
function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$LogPath, [switch]$AllowFailure) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & $FilePath @Arguments 2>&1; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
    if ($LogPath) { $output | Set-Content -LiteralPath $LogPath -Encoding utf8 }
    if ($code -ne 0 -and -not $AllowFailure) { throw "$FilePath failed with exit code $code" }
    [pscustomobject]@{ Lines = @($output | ForEach-Object { "$_" }); ExitCode = $code }
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..\..")).Path
if (-not ((Invoke-Native $AdbPath @("devices") $null).Lines -match "^$([regex]::Escape($DeviceSerial))\s+device$")) { throw "USB device is offline" }
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else { Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-qnn-detailed-operator-profile-r0-$timestamp" }
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null

$deviceRoot = "/data/local/tmp/ba_qairt_htp_r0"
$deviceModelRoot = "$deviceRoot/dav2-metric/518x686"
$deviceOutput = "$deviceModelRoot/output-detailed-operator-r0-$timestamp"
$thermalBeforeLine = (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "dumpsys", "thermalservice") $null).Lines |
    Where-Object { $_ -match '^Thermal Status:\s*(\d+)' } | Select-Object -First 1
$thermalBefore = if ($thermalBeforeLine -match '^Thermal Status:\s*(\d+)') { [int]$Matches[1] } else { $null }
$deviceCommand = "cd $deviceModelRoot && export LD_LIBRARY_PATH=$deviceRoot && export ADSP_LIBRARY_PATH=$deviceRoot\;/system/lib/rfsa/adsp\;/system/vendor/lib/rfsa/adsp\;/dsp && " +
    "$deviceRoot/qnn-net-run --backend $deviceRoot/libQnnHtp.so --model $deviceRoot/libQnnModelDlc.so " +
    "--dlc_path output/model-sm8650-cached.dlc --input_list input-list-clean-native-f16.txt " +
    "--output_dir $deviceOutput --use_native_input_files --use_native_output_files " +
    "--profiling_level detailed --perf_profile sustained_high_performance --duration $DurationSeconds " +
    "--keep_num_outputs 1 --log_level info"
$netRun = Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", $deviceCommand) (Join-Path $artifactRoot "qnn-net-run.txt") -AllowFailure
if ($netRun.ExitCode -ne 0) { throw "detailed qnn-net-run failed" }

$profileLog = Join-Path $artifactRoot "qnn-profiling-data-detailed.log"
$metadata = Join-Path $artifactRoot "execution-metadata.yaml"
(Invoke-Native $AdbPath @("-s", $DeviceSerial, "pull", "$deviceOutput/qnn-profiling-data_0.log", $profileLog) (Join-Path $artifactRoot "pull-profile.txt")).Lines | Out-Null
(Invoke-Native $AdbPath @("-s", $DeviceSerial, "pull", "$deviceOutput/execution_metadata.yaml", $metadata) (Join-Path $artifactRoot "pull-metadata.txt")).Lines | Out-Null

$viewer = Join-Path $QairtRoot "bin\x86_64-windows-msvc\qnn-profile-viewer.exe"
$reader = Join-Path $QairtRoot "lib\x86_64-windows-msvc\QnnHtpProfilingReader.dll"
$csv = Join-Path $artifactRoot "qnn-detailed-operator-profile.csv"
$view = Invoke-Native $viewer @("--input_log", $profileLog, "--reader", $reader, "--output", $csv) (Join-Path $artifactRoot "qnn-profile-viewer.txt") -AllowFailure
if ($view.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $csv)) { throw "QNN HTP profile viewer failed" }
$thermalAfterLine = (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "dumpsys", "thermalservice") $null).Lines |
    Where-Object { $_ -match '^Thermal Status:\s*(\d+)' } | Select-Object -First 1
$thermalAfter = if ($thermalAfterLine -match '^Thermal Status:\s*(\d+)') { [int]$Matches[1] } else { $null }

[ordered]@{
    schema = "blindassist_dav2_qnn_detailed_operator_profile_r0_bundle"
    generated_at = (Get-Date).ToString("o")
    device_serial = $DeviceSerial
    device_model = ((Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "getprop", "ro.product.model") $null).Lines -join "").Trim()
    device_soc = ((Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "getprop", "ro.soc.model") $null).Lines -join "").Trim()
    duration_seconds = $DurationSeconds
    profiling_level = "detailed"
    profiling_semantics = "per-op timing; profiling overhead means not an app latency measurement"
    performance_profile = "sustained_high_performance"
    thermal_status_before = $thermalBefore
    thermal_status_after = $thermalAfter
    cached_dlc_path = "$deviceModelRoot/output/model-sm8650-cached.dlc"
    input_list = "$deviceModelRoot/input-list-clean-native-f16.txt"
    device_output = $deviceOutput
    qnn_profile_log_sha256 = (Get-FileHash -LiteralPath $profileLog -Algorithm SHA256).Hash
    qnn_profile_csv_sha256 = (Get-FileHash -LiteralPath $csv -Algorithm SHA256).Hash
    execution_metadata_sha256 = (Get-FileHash -LiteralPath $metadata -Algorithm SHA256).Hash
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $artifactRoot "result.json") -Encoding utf8
"artifact_root=$artifactRoot"
"profile_csv=$csv"
