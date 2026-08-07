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
& (Join-Path $PSScriptRoot "diagnostics\depthart\run_qnn_native_cached_context_r0.ps1") @PSBoundParameters
exit $LASTEXITCODE
