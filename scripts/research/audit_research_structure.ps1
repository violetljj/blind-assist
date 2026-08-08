[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [switch]$Json,
    [ValidateSet('all','governance','deployment','diagnostics','current','platform','archive','support')]
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
$familyPath = Join-Path $researchRoot 'module_families.json'
if (-not (Test-Path -LiteralPath $familyPath -PathType Leaf)) { throw "Missing research family manifest: $familyPath" }
$families = Get-Content -LiteralPath $familyPath -Raw -Encoding UTF8 | ConvertFrom-Json
$repoFiles = @(
    & git -C $root ls-files --cached --others --exclude-standard -- scripts/research |
        ForEach-Object { $_.Replace('\', '/') } |
        Sort-Object -Unique
)
$familyCounts = [ordered]@{}
foreach ($family in @($families.family_order)) { $familyCounts[$family] = 0 }
$unclassifiedModules = @()
$moduleRows = foreach ($dir in Get-ChildItem -LiteralPath $researchRoot -Directory | Sort-Object Name) {
    $readme = Join-Path $dir.FullName 'README.md'; $missing = @()
    if (Test-Path -LiteralPath $readme -PathType Leaf) {
        $text = Get-Content -LiteralPath $readme -Raw -Encoding UTF8
        foreach ($marker in @('状态：', '## 稳定 Interface', '## 输出', '## 安全边界', '## 停止条件', 'artifacts.local/')) { if (-not $text.Contains($marker)) { $missing += $marker } }
    } else { $missing = @('README.md') }
    $familyMatches = @(
        foreach ($family in @($families.family_order)) {
            foreach ($pattern in @($families.families.$family.patterns)) {
                if ($dir.Name -match [string]$pattern) { $family; break }
            }
        }
    )
    $familyName = if ($familyMatches.Count -eq 1) { [string]$familyMatches[0] } else { 'UNCLASSIFIED' }
    if ($familyName -eq 'UNCLASSIFIED') { $unclassifiedModules += $dir.Name } else { $familyCounts[$familyName]++ }
    $dynamicTruth = if ($familyName -eq 'UNCLASSIFIED') { $null } else { [string]$families.families.$familyName.dynamic_truth }
    $modulePrefix = "scripts/research/$($dir.Name)/"
    [pscustomobject]@{
        module=$dir.Name
        family=$familyName
        dynamic_truth=$dynamicTruth
        readme=(Test-Path $readme -PathType Leaf)
        file_count=@($repoFiles | Where-Object { $_.StartsWith($modulePrefix, [StringComparison]::OrdinalIgnoreCase) }).Count
        missing_contract_markers=@($missing)
    }
}
$roleCounts = [ordered]@{}; $supportFiles = @()
foreach ($candidateRole in @($roles.role_order)) { $roleCounts[$candidateRole] = 0 }
$hftfPrefix = 'scripts/research/hftf/'
$hftfFiles = @($repoFiles | Where-Object { $_.StartsWith($hftfPrefix, [StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { $_.Substring($hftfPrefix.Length) })
foreach ($relative in $hftfFiles) {
    $matched = $null
    foreach ($candidateRole in @($roles.role_order)) { foreach ($pattern in @($roles.roles.$candidateRole.patterns)) { if ($relative -match $pattern) { $matched=$candidateRole; break } }; if ($matched) { break } }
    if (-not $matched) { $matched='unmatched' }; if (-not $roleCounts.Contains($matched)) { $roleCounts[$matched]=0 }; $roleCounts[$matched]++
    if ($matched -eq 'support') { $supportFiles += $relative }
}
$selectedFiles = if ($Role -eq 'all') { @() } else {
    $hftfFiles | ForEach-Object {
        $relative = $_
        $matched = $null
        foreach ($candidateRole in @($roles.role_order)) {
            foreach ($pattern in @($roles.roles.$candidateRole.patterns)) {
                if ($relative -match $pattern) { $matched=$candidateRole; break }
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
    family_counts=$familyCounts
    unclassified_modules=@($unclassifiedModules)
    hftf_role_counts=$roleCounts
    selected_role=$Role
    selected_files=@($selectedFiles)
    hftf_support_file_count=$supportFiles.Count
    hftf_support_files=if ($Role -eq 'all') { @($supportFiles | Select-Object -First ([Math]::Max(0, $MaxFiles))) } else { @() }
    next_action=if($unclassifiedModules.Count){'Classify unregistered research modules.'}elseif($supportFiles.Count){'Classify support files in bounded batches.'}else{'No support files remain.'}
}
if ($Json) { $report | ConvertTo-Json -Depth 8; exit 0 }
Write-Host "Research structure audit: $($report.research_module_count) module(s)"
$roleCounts.GetEnumerator() | Format-Table Name,Value -AutoSize
$familyCounts.GetEnumerator() | Format-Table Name,Value -AutoSize
if ($Role -eq 'all') { Write-Host "HFTF support files: $($supportFiles.Count)" } else { Write-Host "Selected role: $Role" }
@($selectedFiles) | ForEach-Object { Write-Host " - $_" }
if ($Role -eq 'all') { @($supportFiles | Select-Object -First ([Math]::Max(0, $MaxFiles))) | ForEach-Object { Write-Host " - $_" } }
Write-Host "Next action: $($report.next_action)"
