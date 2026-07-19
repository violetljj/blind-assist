param(
    [string]$ApkPath = "app\build\outputs\apk\debug\app-debug.apk",
    [string]$ArchiveDir = "E:\linnan\blind-assist-apk-archive\apks",
    [switch]$Milestone,
    [string]$MilestoneDir = "releases\apk"
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $PSScriptRoot "..\$Path")
}

function Get-VersionName {
    $gradleFile = Join-Path $PSScriptRoot "..\app\build.gradle.kts"
    $line = Select-String -LiteralPath $gradleFile -Pattern 'versionName\s*=\s*"([^"]+)"' | Select-Object -First 1
    if ($line -and $line.Matches[0].Groups[1].Value) {
        return $line.Matches[0].Groups[1].Value
    }
    return "unknown"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$resolvedApk = Resolve-RepoPath $ApkPath
if (-not (Test-Path -LiteralPath $resolvedApk)) {
    throw "APK not found: $resolvedApk"
}
$resolvedApk = (Resolve-Path -LiteralPath $resolvedApk).Path

New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
$versionName = Get-VersionName
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$destName = "BlindAssist-v$versionName-debug-$timestamp.apk"
$archivePath = Join-Path $ArchiveDir $destName
Copy-Item -LiteralPath $resolvedApk -Destination $archivePath -Force

$hash = Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath
$manifest = Join-Path (Split-Path -Parent $ArchiveDir) "APK_ARCHIVE_MANIFEST.csv"
if (-not (Test-Path -LiteralPath $manifest)) {
    "Timestamp,FileName,SizeBytes,SHA256,SourcePath" | Out-File -FilePath $manifest -Encoding utf8
}
$size = (Get-Item -LiteralPath $archivePath).Length
"$timestamp,$destName,$size,$($hash.Hash),$resolvedApk" | Add-Content -Path $manifest -Encoding utf8

$result = [ordered]@{
    archivePath = $archivePath
    sizeBytes = $size
    sha256 = $hash.Hash
    manifest = $manifest
    milestonePath = $null
}

if ($Milestone) {
    $resolvedMilestoneDir = Resolve-RepoPath $MilestoneDir
    New-Item -ItemType Directory -Force -Path $resolvedMilestoneDir | Out-Null
    $milestonePath = Join-Path $resolvedMilestoneDir ("{0}.json" -f [System.IO.Path]::GetFileNameWithoutExtension($destName))
    [pscustomobject]@{
        schema = 'blindassist_apk_receipt_v1'
        archived_at = (Get-Date).ToString('o')
        file_name = $destName
        size_bytes = $hashSize
        sha256 = $hash.Hash
        external_archive_path = $archivePath
        source_path = $resolvedApk
    } | ConvertTo-Json | Set-Content -LiteralPath $milestonePath -Encoding utf8
    $result.milestoneReceiptPath = $milestonePath
}

$result | ConvertTo-Json -Depth 3
