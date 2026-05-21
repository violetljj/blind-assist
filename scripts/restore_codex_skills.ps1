param(
    [string]$CodexHome = (Join-Path $env:USERPROFILE ".codex"),
    [string]$Snapshot = "",
    [switch]$NoBackup
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Snapshot)) {
    $Snapshot = Join-Path $PSScriptRoot "..\codex\skills-snapshot\codex-skills-20260522.zip"
}

$snapshotPath = (Resolve-Path -LiteralPath $Snapshot).Path
$codexHomePath = [System.IO.Path]::GetFullPath($CodexHome)
$skillsPath = Join-Path $codexHomePath "skills"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupPath = Join-Path $codexHomePath "skills-backup-$timestamp"

Write-Host "Codex home: $codexHomePath"
Write-Host "Skills snapshot: $snapshotPath"

New-Item -ItemType Directory -Force -Path $codexHomePath | Out-Null

if ((Test-Path -LiteralPath $skillsPath) -and -not $NoBackup) {
    Write-Host "Existing skills folder found. Creating backup: $backupPath"
    Copy-Item -LiteralPath $skillsPath -Destination $backupPath -Recurse -Force
}

$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if ($tar) {
    Write-Host "Restoring with tar.exe..."
    & $tar.Source -xf $snapshotPath -C $codexHomePath
} else {
    Write-Host "tar.exe not found. Restoring with Expand-Archive..."
    Expand-Archive -LiteralPath $snapshotPath -DestinationPath $codexHomePath -Force
}

$skillDirs = Get-ChildItem -LiteralPath $skillsPath -Directory -Force
$skillFiles = Get-ChildItem -LiteralPath $skillsPath -Recurse -Force -File

Write-Host "Restore complete."
Write-Host "Skill folders: $($skillDirs.Count)"
Write-Host "Skill files: $($skillFiles.Count)"
Write-Host "Restart Codex to reload restored skills."
