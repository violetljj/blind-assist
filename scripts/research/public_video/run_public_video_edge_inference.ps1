param(
    [Parameter(Mandatory = $true)][string]$SilverManifest,
    [Parameter(Mandatory = $true)][string]$SourceManifest,
    [Parameter(Mandatory = $true)][string]$SourceImagesDir,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$AdbPath = "E:\codex-tools\tools\android-sdk\platform-tools\adb.exe",
    [ValidateSet("current", "center_near_strict", "side_near_strict", "animal_aware_candidate", "approaching_center_person_candidate")][string]$RiskConfig = "current",
    [switch]$RemoveConflictingInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

function Resolve-RepoPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Invoke-Native([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $File $($Arguments -join ' ')" }
}

$silver = Resolve-RepoPath $SilverManifest
$source = Resolve-RepoPath $SourceManifest
$images = Resolve-RepoPath $SourceImagesDir
$output = Resolve-RepoPath $OutputRoot
$adb = Resolve-RepoPath $AdbPath
$python = "E:\codex-tools\bin\blindassist-python.cmd"
$dataset = Join-Path $output "device_set"
$pulled = Join-Path $output "device_artifacts"
$expectedSilverSha = (Get-FileHash -LiteralPath $silver -Algorithm SHA256).Hash.ToLowerInvariant()

if (-not (Test-Path -LiteralPath $silver) -or -not (Test-Path -LiteralPath $source) -or -not (Test-Path -LiteralPath $images)) {
    throw "Silver manifest, source manifest, and source images must exist."
}
if (Test-Path -LiteralPath $output) { throw "Refusing to overwrite output root: $output" }
if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $adb)) { throw "Required Python or ADB tool is missing." }

$devices = & $adb devices
$online = @($devices | Where-Object { $_ -match "^\S+\s+device$" } | ForEach-Object { ($_ -split '\s+')[0] })
if ($online.Count -ne 1) { throw "Exactly one authorized ADB device is required; found $($online.Count)." }
$serial = $online[0]

Push-Location $repoRoot
try {
    Invoke-Native $python @(
        "scripts\run_research_tool.py", "public-video", "build_public_video_edge_inference_set.py",
        "--silver-manifest", $silver,
        "--source-manifest", $source,
        "--source-images-dir", $images,
        "--output-root", $dataset
    )
    if ($RemoveConflictingInstall) {
        # Explicitly opt-in only: these remove existing local packages and their app data.
        Invoke-Native $adb @("-s", $serial, "uninstall", "com.linnan.blindassist")
        Invoke-Native $adb @("-s", $serial, "uninstall", "com.linnan.blindassist.benchmark")
    }
    $env:JAVA_HOME = "E:\codex-tools\jdk-17"
    $env:Path = "$env:JAVA_HOME\bin;$env:Path"
    Invoke-Native ".\gradlew.bat" @(
        ":device-benchmark:connectedDebugAndroidTest",
        "-PpublicVideoInferenceDir=$($dataset.Replace('\', '/'))",
        "-Pandroid.testInstrumentationRunnerArguments.class=com.linnan.blindassist.benchmark.PublicVideoEdgeInferenceTest",
        "-Pandroid.testInstrumentationRunnerArguments.publicVideoInferenceRequired=true",
        "-Pandroid.testInstrumentationRunnerArguments.publicVideoRiskConfig=$RiskConfig",
        "-Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true",
        "--no-daemon",
        "--console=plain"
    )
    New-Item -ItemType Directory -Force -Path $pulled | Out-Null
    Invoke-Native $adb @("-s", $serial, "pull", "/sdcard/Android/data/com.linnan.blindassist/files/public-video-edge-inference", $pulled)
    $edgeReport = @(
        Get-ChildItem -LiteralPath $pulled -Filter edge_events.json -Recurse |
            ForEach-Object {
                $report = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                if ($report.silver_manifest_sha256 -eq $expectedSilverSha -and $report.risk_config -eq $RiskConfig) {
                    [pscustomobject]@{ File = $_; CreatedAtUtc = $report.created_at_utc }
                }
            } |
            Sort-Object CreatedAtUtc -Descending
    ) | Select-Object -First 1
    if ($null -eq $edgeReport) { throw "Device test passed but did not produce edge_events.json." }
    Invoke-Native $python @(
        "scripts\run_research_tool.py", "public-video", "compare_public_silver_to_edge_events.py",
        "--silver-manifest", $silver,
        "--source-manifest", $source,
        "--edge-report", $edgeReport.File.FullName,
        "--output", (Join-Path $output "edge_comparison.json")
    )
    Write-Host "Public-video silver comparison: $(Join-Path $output 'edge_comparison.json')"
} finally {
    Pop-Location
}
