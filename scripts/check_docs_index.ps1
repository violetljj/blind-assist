param(
    [string]$DocsRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs')
)

$ErrorActionPreference = 'Stop'
$docsRootPath = (Resolve-Path -LiteralPath $DocsRoot).Path
$indexPath = Join-Path $docsRootPath 'README.md'

if (-not (Test-Path -LiteralPath $indexPath)) {
    throw "Documentation index is missing: $indexPath"
}

$indexText = Get-Content -LiteralPath $indexPath -Raw -Encoding utf8
$failures = New-Object System.Collections.Generic.List[string]

if (-not (Test-Path -LiteralPath (Join-Path $docsRootPath 'PROJECT_STATE.md') -PathType Leaf)) {
    $failures.Add('Cold-start navigation file is missing: PROJECT_STATE.md')
}
elseif ($indexText -notmatch [regex]::Escape('PROJECT_STATE.md')) {
    $failures.Add('Documentation index does not link cold-start navigation: PROJECT_STATE.md')
}

Get-ChildItem -LiteralPath $docsRootPath -File -Filter *.md |
    Where-Object { $_.Name -ne 'README.md' } |
    Sort-Object Name |
    ForEach-Object {
        if ($indexText -notmatch [regex]::Escape($_.Name)) {
            $failures.Add("Top-level documentation file is not indexed: $($_.Name)")
        }
    }

$researchRoot = Join-Path $docsRootPath 'research'
if (Test-Path -LiteralPath $researchRoot -PathType Container) {
    $researchIndexPath = Join-Path $researchRoot 'README.md'
    $requiredResearchEntries = @(
        'README.md',
        'ALGORITHM_RESEARCH_CURRENT.md',
        'DATA_RESEARCH_CURRENT.md',
        'SYSTEM_RESEARCH_CURRENT.md'
    )
    foreach ($entry in $requiredResearchEntries) {
        $entryPath = Join-Path $researchRoot $entry
        if (-not (Test-Path -LiteralPath $entryPath -PathType Leaf)) {
            $failures.Add("Required research entry is missing: research/$entry")
        }
    }

    if (Test-Path -LiteralPath $researchIndexPath -PathType Leaf) {
        $researchIndexText = Get-Content -LiteralPath $researchIndexPath -Raw -Encoding utf8
        foreach ($entry in @(
            'ALGORITHM_RESEARCH_CURRENT.md',
            'DATA_RESEARCH_CURRENT.md',
            'SYSTEM_RESEARCH_CURRENT.md'
        )) {
            if ($researchIndexText -notmatch [regex]::Escape($entry)) {
                $failures.Add("Research index does not link required category entry: research/$entry")
            }
        }
    }

    Get-ChildItem -LiteralPath $researchRoot -Directory |
        Sort-Object Name |
        ForEach-Object {
            $domainReadme = Join-Path $_.FullName 'README.md'
            if (-not (Test-Path -LiteralPath $domainReadme -PathType Leaf)) {
                $failures.Add("Research documentation domain lacks a README index: research/$($_.Name)/README.md")
            }
            $domainIndexTarget = "research/$($_.Name)/README.md"
            if ($indexText -notmatch [regex]::Escape($domainIndexTarget)) {
                $failures.Add("Research documentation domain is not linked from docs/README.md: $domainIndexTarget")
            }
        }
}

$linkPattern = '\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)'
$linkSourcePaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($sourcePath in @(
    $indexPath,
    (Join-Path $researchRoot 'README.md'),
    (Join-Path $researchRoot 'ALGORITHM_RESEARCH_CURRENT.md'),
    (Join-Path $researchRoot 'DATA_RESEARCH_CURRENT.md'),
    (Join-Path $researchRoot 'SYSTEM_RESEARCH_CURRENT.md')
)) {
    if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
        [void]$linkSourcePaths.Add([IO.Path]::GetFullPath($sourcePath))
    }
}

# Validate the complete operating surface, not just the top-level indexes.
# Historical archive and monthly history preserve old paths verbatim; current
# docs, every route README, and non-archive protocols must remain navigable.
foreach ($candidate in Get-ChildItem -LiteralPath $docsRootPath -Recurse -File -Filter *.md) {
    $relativeCandidate = [IO.Path]::GetRelativePath($docsRootPath, $candidate.FullName).Replace('\', '/')
    if ($relativeCandidate.StartsWith('history/', [StringComparison]::OrdinalIgnoreCase) -or
        $relativeCandidate -match '(^|/)archive/') {
        continue
    }
    $head = (Get-Content -LiteralPath $candidate.FullName -TotalCount 16 -Encoding utf8) -join "`n"
    $isCurrent = $head -match '(?im)^状态：\s*`?current'
    $isRouteIndex = $candidate.Name -eq 'README.md'
    $isProtocol = $candidate.Name -match 'PROTOCOL'
    if ($isCurrent -or $isRouteIndex -or $isProtocol) {
        [void]$linkSourcePaths.Add($candidate.FullName)
    }
}

$validatedLocalLinkCount = 0
foreach ($sourcePath in @($linkSourcePaths | Sort-Object)) {
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        continue
    }
    $sourceText = Get-Content -LiteralPath $sourcePath -Raw -Encoding utf8
    foreach ($match in [regex]::Matches($sourceText, $linkPattern)) {
        $target = $match.Groups[1].Value.Trim().Trim('<', '>')
        if ($target -match '^[a-zA-Z][a-zA-Z0-9+.-]*:' -or $target.StartsWith('/')) {
            continue
        }

        $validatedLocalLinkCount++
        $targetPath = Join-Path (Split-Path -Parent $sourcePath) $target
        if (-not (Test-Path -LiteralPath $targetPath)) {
            $relativeSource = [IO.Path]::GetRelativePath($docsRootPath, $sourcePath).Replace('\', '/')
            $failures.Add("Documentation entry link target is missing in ${relativeSource}: $target")
        }
    }
}

if ($failures.Count -gt 0) {
    Write-Host 'Documentation index check failed:'
    foreach ($failure in $failures) {
        Write-Host " - $failure"
    }
    exit 1
}

$researchDomainCount = if (Test-Path -LiteralPath $researchRoot -PathType Container) {
    (Get-ChildItem -LiteralPath $researchRoot -Directory).Count
}
else {
    0
}
Write-Host "Documentation index check passed for $((Get-ChildItem -LiteralPath $docsRootPath -File -Filter *.md).Count - 1) top-level Markdown file(s), $researchDomainCount research domain(s), and $validatedLocalLinkCount authority-surface local link(s)."
