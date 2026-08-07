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
foreach ($sourcePath in @(
    $indexPath,
    (Join-Path $researchRoot 'README.md'),
    (Join-Path $researchRoot 'ALGORITHM_RESEARCH_CURRENT.md'),
    (Join-Path $researchRoot 'DATA_RESEARCH_CURRENT.md'),
    (Join-Path $researchRoot 'SYSTEM_RESEARCH_CURRENT.md')
)) {
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        continue
    }
    $sourceText = Get-Content -LiteralPath $sourcePath -Raw -Encoding utf8
    foreach ($match in [regex]::Matches($sourceText, $linkPattern)) {
        $target = $match.Groups[1].Value
        if ($target -match '^[a-zA-Z][a-zA-Z0-9+.-]*:' -or $target.StartsWith('/')) {
            continue
        }

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
Write-Host "Documentation index check passed for $((Get-ChildItem -LiteralPath $docsRootPath -File -Filter *.md).Count - 1) top-level Markdown file(s) and $researchDomainCount research domain(s)."
