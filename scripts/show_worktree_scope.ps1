[CmdletBinding()]
param([switch]$Details)

$ErrorActionPreference = 'Stop'
$repoRoot = (& git rev-parse --show-toplevel 2>$null | Select-Object -First 1).Trim()
if (-not $repoRoot) { throw 'Run inside the BlindAssist Git checkout.' }

function Get-WorkLane([string]$Path) {
    switch -Regex ($Path) {
        '^research/active/dtr-r0/|^tools/run_dtr_' { return 'DTR' }
        '^research/active/l10-r0/|^research/active/(named-poi-v1|open-world-poi-grounding)/' { return 'L10-POI' }
        '^apps/|^app/|^core/|^feature/|^scripts/run_(android|device)|^docs/(APK_ARCHIVE|BLINDASSIST_EVALSET|DETECTOR_BENCHMARK|DEVICE_REGRESSION)' { return 'Android-device' }
        '^data/|^tools/data/|^docs/(RESOURCE_FABRIC|asset-management/)' { return 'Data-assets' }
        '^AGENTS\.md$|^README\.md$|^docs/(PROJECT_STATE|CURRENT_DECISION|DOCUMENT_GOVERNANCE|README)\.md$|^scripts/(check_docs_index|check_project_structure|show_worktree_scope)\.ps1$' { return 'Governance' }
        default { return 'Other' }
    }
}

$authorityPaths = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
@(
    'AGENTS.md', 'README.md', 'docs/README.md', 'docs/PROJECT_STATE.md',
    'docs/CURRENT_DECISION.md', 'docs/DOCUMENT_GOVERNANCE.md',
    'research/active/l10-r0/CURRENT.md',
    'research/active/dtr-r0/CURRENT.md'
) | ForEach-Object { $null = $authorityPaths.Add($_) }

$statusLines = @(
    & git -C $repoRoot -c core.quotePath=false status --porcelain=v1 --untracked-files=normal
)
if ($LASTEXITCODE -ne 0) { throw 'git status failed.' }

$entries = foreach ($line in $statusLines) {
    if ($line.Length -lt 4) { continue }
    $state = $line.Substring(0, 2)
    $path = $line.Substring(3)
    if ($path.Contains(' -> ')) { $path = ($path -split ' -> ')[-1] }
    $untracked = $state -eq '??'
    $staged = -not $untracked -and $state[0] -ne ' '
    $unstaged = -not $untracked -and $state[1] -ne ' '
    $kind = if ($untracked) {
        'untracked'
    } elseif ($staged -and $unstaged) {
        'staged+unstaged'
    } elseif ($staged) {
        'staged'
    } else {
        'modified'
    }
    [pscustomobject]@{
        Lane = Get-WorkLane $path
        Kind = $kind
        Path = $path
        Staged = $staged
        Unstaged = $unstaged
        Untracked = $untracked
        Authority = $authorityPaths.Contains($path)
    }
}

$branch = (& git -C $repoRoot rev-parse --abbrev-ref HEAD).Trim()
$upstreamLines = @(
    & git -C $repoRoot rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>$null
)
$upstream = if ($LASTEXITCODE -eq 0 -and $upstreamLines.Count) {
    $upstreamLines[0].Trim()
} else {
    '(none)'
}
$ahead = '?'
$behind = '?'
if ($upstream -ne '(none)') {
    $counts = ((& git -C $repoRoot rev-list --left-right --count "HEAD...$upstream").Trim() -split '\s+')
    if ($LASTEXITCODE -eq 0 -and $counts.Count -eq 2) {
        $ahead = $counts[0]
        $behind = $counts[1]
    }
}

Write-Host "Worktree scope: $branch -> $upstream (ahead=$ahead, behind=$behind)"
if (-not $entries) {
    Write-Host 'Worktree is clean.'
    exit 0
}

$summary = foreach ($group in ($entries | Group-Object Lane | Sort-Object Name)) {
    [pscustomobject]@{
        Lane = $group.Name
        Entries = $group.Count
        Staged = @($group.Group | Where-Object Staged).Count
        Modified = @($group.Group | Where-Object Unstaged).Count
        Untracked = @($group.Group | Where-Object Untracked).Count
        AuthorityWip = @($group.Group | Where-Object Authority).Count
    }
}
$summary | Format-Table -AutoSize

$authorityCount = @($entries | Where-Object Authority).Count
Write-Host "Total entries: $($entries.Count); authority WIP: $authorityCount"

if ($Details) {
    $entries |
        Sort-Object Lane, Path |
        Select-Object Lane, Kind, Authority, Path |
        Format-Table -AutoSize -Wrap
}
