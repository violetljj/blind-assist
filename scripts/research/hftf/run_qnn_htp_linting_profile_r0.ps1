param(
    [string]$AdbPath = "E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe",
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$QairtRoot = "E:\codex-tools\qairt\2.47.0.260601",
    [int]$DurationSeconds = 5,
    [string]$OutputRoot
)
& (Join-Path $PSScriptRoot "diagnostics\depthart\run_qnn_htp_linting_profile_r0.ps1") @PSBoundParameters
exit $LASTEXITCODE
