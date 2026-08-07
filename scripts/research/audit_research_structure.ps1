[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [switch]$Json,
    [ValidateSet('all','deployment','diagnostics','current','archive','support')]
    [string]$Role = 'all',
    [int]$MaxFiles = 20
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $RepoRoot).Path
$researchRoot = Join-Path $root 'scripts/research'
$hftfRoot = Join-Path $researchRoot 'hftf'
$rolesPath = Join-Path $hftfRoot 'roles.json'
if (-not (Test-Path -LiteralPath $rolesPath -PathType Leaf)) { throw "Missing HFTF role manifest: $rolesPath" }
$roles = Get-Content -LiteralPath $rolesPath -Raw -Encoding UTF8 | ConvertFrom-Json
$moduleRows = foreach ($dir in Get-ChildItem -LiteralPath $researchRoot -Directory | Sort-Object Name) {
    $readme = Join-Path $dir.FullName 'README.md'; $missing = @()
    if (Test-Path -LiteralPath $readme -PathType Leaf) {
        $text = Get-Content -LiteralPath $readme -Raw -Encoding UTF8
        foreach ($marker in @('状态：', '## 稳定 Interface', '## 输出', '## 安全边界', '## 停止条件', 'artifacts.local/')) { if (-not $text.Contains($marker)) { $missing += $marker } }
    } else { $missing = @('README.md') }
    [pscustomobject]@{ module=$dir.Name; readme=(Test-Path $readme -PathType Leaf); file_count=@(Get-ChildItem $dir.FullName -Recurse -File).Count; missing_contract_markers=@($missing) }
}
$roleCounts = [ordered]@{}; $supportFiles = @()
foreach ($role in @($roles.role_order)) { $roleCounts[$role] = 0 }
foreach ($file in Get-ChildItem -LiteralPath $hftfRoot -Recurse -File) {
    $relative = [IO.Path]::GetRelativePath($hftfRoot, $file.FullName).Replace('\', '/'); $matched = $null
    if ($relative -match '(^|/)(__pycache__|\.pytest_cache)/' -or $relative -match '\.pyc$') { continue }
    foreach ($role in @($roles.role_order)) { foreach ($pattern in @($roles.roles.$role.patterns)) { if ($relative -match $pattern) { $matched=$role; break } }; if ($matched) { break } }
    if (-not $matched) { $matched='unmatched' }; if (-not $roleCounts.Contains($matched)) { $roleCounts[$matched]=0 }; $roleCounts[$matched]++
    if ($matched -eq 'support') { $supportFiles += $relative }
}
$selectedFiles = if ($Role -eq 'all') { @() } else {
    Get-ChildItem -LiteralPath $hftfRoot -Recurse -File | ForEach-Object {
        $relative = [IO.Path]::GetRelativePath($hftfRoot, $_.FullName).Replace('\', '/')
        if ($relative -match '(^|/)(__pycache__|\.pytest_cache)/' -or $relative -match '\.pyc$') { return }
        $matched = $null
        foreach ($candidate in @($roles.role_order)) {
            foreach ($pattern in @($roles.roles.$candidate.patterns)) {
                if ($relative -match $pattern) { $matched=$candidate; break }
            }
            if ($matched) { break }
        }
        if ($matched -eq $Role) { $relative }
    } | Sort-Object | Select-Object -First ([Math]::Max(0, $MaxFiles))
}
$report = [pscustomobject]@{
    repo_root=$root
    research_module_count=@($moduleRows).Count
    modules=@($moduleRows)
    hftf_role_counts=$roleCounts
    selected_role=$Role
    selected_files=@($selectedFiles)
    hftf_support_file_count=$supportFiles.Count
    hftf_support_files=if ($Role -eq 'all') { @($supportFiles | Select-Object -First ([Math]::Max(0, $MaxFiles))) } else { @() }
    next_action=if($supportFiles.Count){'Classify support files in bounded batches.'}else{'No support files remain.'}
}
if ($Json) { $report | ConvertTo-Json -Depth 8; exit 0 }
Write-Host "Research structure audit: $($report.research_module_count) module(s)"
$roleCounts.GetEnumerator() | Format-Table Name,Value -AutoSize
if ($Role -eq 'all') { Write-Host "HFTF support files: $($supportFiles.Count)" } else { Write-Host "Selected role: $Role" }
@($selectedFiles) | ForEach-Object { Write-Host " - $_" }
if ($Role -eq 'all') { @($supportFiles | Select-Object -First ([Math]::Max(0, $MaxFiles))) | ForEach-Object { Write-Host " - $_" } }
Write-Host "Next action: $($report.next_action)"
