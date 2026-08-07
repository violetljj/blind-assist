param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$QairtRoot = "E:\codex-tools\qairt\2.47.0.260601",
    [int]$DurationSeconds = 5,
    [string]$OutputRoot
)
$ErrorActionPreference = "Stop"
if ($DurationSeconds -lt 1 -or $DurationSeconds -gt 30) { throw "DurationSeconds must be in 1..30" }
function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$LogPath, [switch]$AllowFailure) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & $FilePath @Arguments 2>&1; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
    if ($LogPath) { $output | Set-Content -LiteralPath $LogPath -Encoding utf8 }
    if ($code -ne 0 -and -not $AllowFailure) { throw "$FilePath failed with exit code $code" }
    [pscustomobject]@{ Lines = @($output | ForEach-Object { "$_" }); ExitCode = $code }
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not ((Invoke-Native $AdbPath @("devices") $null).Lines -match "^$([regex]::Escape($DeviceSerial))\s+device$")) { throw "USB device is offline" }
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = if ($OutputRoot) { [IO.Path]::GetFullPath($OutputRoot) } else { Join-Path $repoRoot "artifacts.local\evidence\hftf\dav2-qnn-htp-linting-profile-r0-$timestamp" }
if (Test-Path -LiteralPath $artifactRoot) { throw "output already exists" }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null

$deviceRoot = "/data/local/tmp/ba_qairt_htp_r0"
$deviceModelRoot = "$deviceRoot/dav2-metric/518x686"
$deviceHtpConfig = "$deviceModelRoot/qnn-htp-linting-r0.json"
$deviceBackendConfig = "$deviceModelRoot/qnn-htp-linting-backend-r0.json"
$htpConfig = Join-Path $PSScriptRoot "diagnostics\depthart\qnn_htp_linting_config_r0.json"
$backendConfig = Join-Path $PSScriptRoot "diagnostics\depthart\qnn_htp_linting_backend_extensions_r0.json"
(Invoke-Native $AdbPath @("-s", $DeviceSerial, "push", $htpConfig, $deviceHtpConfig) (Join-Path $artifactRoot "push-htp-config.txt")).Lines | Out-Null
(Invoke-Native $AdbPath @("-s", $DeviceSerial, "push", $backendConfig, $deviceBackendConfig) (Join-Path $artifactRoot "push-backend-config.txt")).Lines | Out-Null

$thermalBeforeLine = (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "dumpsys", "thermalservice") $null).Lines |
    Where-Object { $_ -match '^Thermal Status:\s*(\d+)' } | Select-Object -First 1
$thermalBefore = if ($thermalBeforeLine -match '^Thermal Status:\s*(\d+)') { [int]$Matches[1] } else { $null }
$deviceOutput = "$deviceModelRoot/output-htp-linting-r0-$timestamp"
$deviceCommand = "cd $deviceModelRoot && export LD_LIBRARY_PATH=$deviceRoot && export ADSP_LIBRARY_PATH=$deviceRoot\;/system/lib/rfsa/adsp\;/system/vendor/lib/rfsa/adsp\;/dsp && " +
    "$deviceRoot/qnn-net-run --backend $deviceRoot/libQnnHtp.so --model $deviceRoot/libQnnModelDlc.so " +
    "--dlc_path output/model-sm8650-cached.dlc --input_list input-list-clean-native-f16.txt " +
    "--output_dir $deviceOutput --use_native_input_files --use_native_output_files " +
    "--profiling_level backend --config_file $deviceBackendConfig --perf_profile sustained_high_performance " +
    "--duration $DurationSeconds --keep_num_outputs 1 --log_level info"
$netRun = Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", $deviceCommand) (Join-Path $artifactRoot "qnn-net-run.txt") -AllowFailure
if ($netRun.ExitCode -ne 0) { throw "HTP linting qnn-net-run failed" }

$profileLog = Join-Path $artifactRoot "qnn-profiling-data-htp-linting.log"
$metadata = Join-Path $artifactRoot "execution-metadata.yaml"
(Invoke-Native $AdbPath @("-s", $DeviceSerial, "pull", "$deviceOutput/qnn-profiling-data_0.log", $profileLog) (Join-Path $artifactRoot "pull-profile.txt")).Lines | Out-Null
(Invoke-Native $AdbPath @("-s", $DeviceSerial, "pull", "$deviceOutput/execution_metadata.yaml", $metadata) (Join-Path $artifactRoot "pull-metadata.txt")).Lines | Out-Null
$viewer = Join-Path $QairtRoot "bin\x86_64-windows-msvc\qnn-profile-viewer.exe"
$reader = Join-Path $QairtRoot "lib\x86_64-windows-msvc\QnnHtpProfilingReader.dll"
$output = Join-Path $artifactRoot "qnn-htp-linting-profile.txt"
$view = Invoke-Native $viewer @("--input_log", $profileLog, "--reader", $reader) $output -AllowFailure
if ($view.ExitCode -ne 0) { throw "QNN HTP linting profile viewer failed" }
$thermalAfterLine = (Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "dumpsys", "thermalservice") $null).Lines |
    Where-Object { $_ -match '^Thermal Status:\s*(\d+)' } | Select-Object -First 1
$thermalAfter = if ($thermalAfterLine -match '^Thermal Status:\s*(\d+)') { [int]$Matches[1] } else { $null }

[ordered]@{
    schema = "blindassist_dav2_qnn_htp_linting_profile_r0_bundle"
    generated_at = (Get-Date).ToString("o")
    device_serial = $DeviceSerial
    device_model = ((Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "getprop", "ro.product.model") $null).Lines -join "").Trim()
    device_soc = ((Invoke-Native $AdbPath @("-s", $DeviceSerial, "shell", "getprop", "ro.soc.model") $null).Lines -join "").Trim()
    duration_seconds = $DurationSeconds
    profiling_level = "backend:linting"
    profiling_semantics = "HTP resources and overlap diagnostic; not App latency"
    thermal_status_before = $thermalBefore
    thermal_status_after = $thermalAfter
    htp_config_sha256 = (Get-FileHash -LiteralPath $htpConfig -Algorithm SHA256).Hash
    backend_config_sha256 = (Get-FileHash -LiteralPath $backendConfig -Algorithm SHA256).Hash
    qnn_profile_log_sha256 = (Get-FileHash -LiteralPath $profileLog -Algorithm SHA256).Hash
    qnn_linting_text_sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
    execution_metadata_sha256 = (Get-FileHash -LiteralPath $metadata -Algorithm SHA256).Hash
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $artifactRoot "result.json") -Encoding utf8
"artifact_root=$artifactRoot"
"linting_output=$output"
