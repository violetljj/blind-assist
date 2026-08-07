[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [switch]$Json
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
    foreach ($role in @($roles.role_order)) { foreach ($pattern in @($roles.roles.$role.patterns)) { if ($relative -match $pattern) { $matched=$role; break } }; if ($matched) { break } }
    if (-not $matched) { $matched='unmatched' }; if (-not $roleCounts.Contains($matched)) { $roleCounts[$matched]=0 }; $roleCounts[$matched]++
    if ($matched -eq 'support') { $supportFiles += $relative }
}
$report = [pscustomobject]@{ repo_root=$root; research_module_count=@($moduleRows).Count; modules=@($moduleRows); hftf_role_counts=$roleCounts; hftf_support_file_count=$supportFiles.Count; hftf_support_files=@($supportFiles); next_action=if($supportFiles.Count){'Classify support files in bounded batches.'}else{'No support files remain.'} }
if ($Json) { $report | ConvertTo-Json -Depth 8; exit 0 }
Write-Host "Research structure audit: $($report.research_module_count) module(s)"; $report.modules | Format-Table module,readme,file_count,missing_contract_markers -AutoSize; $roleCounts.GetEnumerator() | Format-Table Name,Value -AutoSize; Write-Host "HFTF support files: $($supportFiles.Count)"; $supportFiles | Select-Object -First 20 | ForEach-Object { Write-Host " - $_" }; Write-Host "Next action: $($report.next_action)"
