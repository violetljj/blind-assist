param(
    [string]$DocsRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'docs')
)

$ErrorActionPreference = 'Stop'
$docsRootPath = (Resolve-Path -LiteralPath $DocsRoot).Path
$repoRootPath = [IO.Path]::GetFullPath((Split-Path -Parent $docsRootPath))
$repoRootPrefix = $repoRootPath.TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
) + [IO.Path]::DirectorySeparatorChar
$indexPath = Join-Path $docsRootPath 'README.md'

if (-not (Test-Path -LiteralPath $indexPath -PathType Leaf)) {
    throw "Documentation index is missing: $indexPath"
}

$indexText = Get-Content -LiteralPath $indexPath -Raw -Encoding utf8
$failures = [Collections.Generic.List[string]]::new()

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
$algorithmCurrentPath = Join-Path $researchRoot 'ALGORITHM_RESEARCH_CURRENT.md'
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
            else {
                $routeLineCount = [IO.File]::ReadAllLines(
                    $domainReadme,
                    [Text.Encoding]::UTF8
                ).Count
                if ($routeLineCount -gt 180) {
                    $failures.Add(
                        "Research route README exceeds the 180-line operating-surface budget: research/$($_.Name)/README.md ($routeLineCount lines)"
                    )
                }
            }
            $domainIndexTarget = "research/$($_.Name)/README.md"
            if ($indexText -notmatch [regex]::Escape($domainIndexTarget)) {
                $failures.Add("Research documentation domain is not linked from docs/README.md: $domainIndexTarget")
            }
        }

    if (Test-Path -LiteralPath $algorithmCurrentPath -PathType Leaf) {
        foreach ($line in [IO.File]::ReadAllLines($algorithmCurrentPath, [Text.Encoding]::UTF8)) {
            if ($line -match '^\|' -and
                $line -notmatch '^\|\s*---' -and
                $line -notmatch '^\|\s*路线\s*\|' -and
                $line.Length -gt 700) {
                $failures.Add(
                    "Algorithm current route row exceeds the 700-character summary budget: $($line.Substring(0, [Math]::Min(80, $line.Length)))..."
                )
            }
        }
    }
}

$linkPattern = '\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)'
$linkSourcePaths = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
foreach ($sourcePath in @(
    $indexPath,
    (Join-Path $researchRoot 'README.md'),
    $algorithmCurrentPath,
    (Join-Path $researchRoot 'DATA_RESEARCH_CURRENT.md'),
    (Join-Path $researchRoot 'SYSTEM_RESEARCH_CURRENT.md')
)) {
    if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
        [void]$linkSourcePaths.Add([IO.Path]::GetFullPath($sourcePath))
    }
}

# Ordinary history preserves source text. Aggregate README files still carry
# navigation responsibility, including when they live under history/archive.
foreach ($candidate in Get-ChildItem -LiteralPath $docsRootPath -Recurse -File -Filter *.md) {
    $relativeCandidate = [IO.Path]::GetRelativePath(
        $docsRootPath,
        $candidate.FullName
    ).Replace('\', '/')
    $head = (Get-Content -LiteralPath $candidate.FullName -TotalCount 16 -Encoding utf8) -join [Environment]::NewLine
    $declaresCurrent = $head -match '(?im)^状态：\s*(?:\x60)?current'
    $isHistoricalSnapshotName = $candidate.Name -match '(?i)(?:^|_)CURRENT_SNAPSHOT_\d{4}-\d{2}-\d{2}\.md$'
    if ($isHistoricalSnapshotName -and $declaresCurrent) {
        $failures.Add("Historical current-snapshot declares current authority: $relativeCandidate")
    }

    $isHistorical = $relativeCandidate.StartsWith(
        'history/',
        [StringComparison]::OrdinalIgnoreCase
    ) -or $relativeCandidate -match '(?i)(^|/)archive/'
    $isAggregateIndex = $candidate.Name -match '(?i)^README(?:[_-].+)?\.md$'
    if ($isHistorical -and -not $isAggregateIndex) {
        continue
    }

    $isRouteIndex = $candidate.Name -eq 'README.md'
    $isProtocol = $candidate.Name -match '(?i)PROTOCOL'
    if ($declaresCurrent -or $isRouteIndex -or $isProtocol -or $isAggregateIndex) {
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
            $relativeSource = [IO.Path]::GetRelativePath(
                $docsRootPath,
                $sourcePath
            ).Replace('\', '/')
            $failures.Add("Documentation entry link target is missing in $($relativeSource): $target")
        }
    }
}

$governedPathPrefixes = @(
    '.github/',
    'app/',
    'apps/',
    'codex/',
    'configs/',
    'core/',
    'docs/',
    'feature/',
    'gradle/',
    'releases/',
    'schemas/',
    'scripts/'
)
$governedRootFiles = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
foreach ($rootFile in @(
    'AGENTS.md',
    'CHANGELOG.md',
    'CONTRIBUTING.md',
    'GOVERNANCE.md',
    'LICENSE',
    'README.md',
    'SECURITY.md',
    'THIRD_PARTY_NOTICES.md',
    'build.gradle.kts',
    'gradle.properties',
    'settings.gradle.kts'
)) {
    [void]$governedRootFiles.Add($rootFile)
}

$jsonPathFieldCount = 0
$validatedJsonPathCount = 0
function Test-GovernedJsonPath {
    param(
        [string]$Value,
        [string]$JsonRelativePath,
        [string]$PropertyName
    )

    $normalized = $Value.Trim().Replace('\', '/')
    if ([string]::IsNullOrWhiteSpace($normalized) -or
        $normalized -match '^[a-zA-Z][a-zA-Z0-9+.-]*://' -or
        [IO.Path]::IsPathRooted($normalized) -or
        $normalized.StartsWith('/') -or
        $normalized.StartsWith('artifacts.local/', [StringComparison]::OrdinalIgnoreCase) -or
        $normalized.StartsWith('test-artifacts.local/', [StringComparison]::OrdinalIgnoreCase) -or
        $normalized.StartsWith('build/', [StringComparison]::OrdinalIgnoreCase) -or
        $normalized -match '(?i)(^|/)build/') {
        return
    }

    $isGoverned = $governedRootFiles.Contains($normalized)
    if (-not $isGoverned) {
        foreach ($prefix in $governedPathPrefixes) {
            if ($normalized.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                $isGoverned = $true
                break
            }
        }
    }
    if (-not $isGoverned) {
        return
    }

    $script:validatedJsonPathCount++
    $resolvedTarget = [IO.Path]::GetFullPath(
        (Join-Path $repoRootPath $normalized.Replace('/', [IO.Path]::DirectorySeparatorChar))
    )
    if (-not (
        $resolvedTarget.Equals($repoRootPath, [StringComparison]::OrdinalIgnoreCase) -or
        $resolvedTarget.StartsWith($repoRootPrefix, [StringComparison]::OrdinalIgnoreCase)
    )) {
        $failures.Add(
            "Governed JSON path escapes the repository in $($JsonRelativePath) [$PropertyName]: $Value"
        )
        return
    }
    if (-not (Test-Path -LiteralPath $resolvedTarget)) {
        $failures.Add(
            "Governed JSON path target is missing in $($JsonRelativePath) [$PropertyName]: $Value"
        )
    }
}

function Visit-JsonNode {
    param(
        [AllowNull()][object]$Node,
        [string]$JsonRelativePath
    )

    if ($null -eq $Node) {
        return
    }
    if ($Node -is [Management.Automation.PSCustomObject]) {
        foreach ($property in $Node.PSObject.Properties) {
            if ($property.Name -match '(?i)^(?:path|.+_path)$' -and
                $property.Value -is [string]) {
                $script:jsonPathFieldCount++
                Test-GovernedJsonPath -Value ([string]$property.Value) -JsonRelativePath $JsonRelativePath -PropertyName $property.Name
            }
            Visit-JsonNode -Node $property.Value -JsonRelativePath $JsonRelativePath
        }
        return
    }
    if ($Node -is [Collections.IDictionary]) {
        foreach ($key in $Node.Keys) {
            $value = $Node[$key]
            if ([string]$key -match '(?i)^(?:path|.+_path)$' -and $value -is [string]) {
                $script:jsonPathFieldCount++
                Test-GovernedJsonPath -Value ([string]$value) -JsonRelativePath $JsonRelativePath -PropertyName ([string]$key)
            }
            Visit-JsonNode -Node $value -JsonRelativePath $JsonRelativePath
        }
        return
    }
    if ($Node -is [Collections.IEnumerable] -and $Node -isnot [string]) {
        foreach ($item in $Node) {
            Visit-JsonNode -Node $item -JsonRelativePath $JsonRelativePath
        }
    }
}

$jsonDocumentCount = 0
foreach ($jsonFile in Get-ChildItem -LiteralPath $docsRootPath -Recurse -File -Filter *.json) {
    $jsonRelativePath = [IO.Path]::GetRelativePath(
        $repoRootPath,
        $jsonFile.FullName
    ).Replace('\', '/')
    $docsRelativePath = [IO.Path]::GetRelativePath(
        $docsRootPath,
        $jsonFile.FullName
    ).Replace('\', '/')
    if ($docsRelativePath.StartsWith('history/', [StringComparison]::OrdinalIgnoreCase) -or
        $docsRelativePath -match '(?i)(^|/)archive/') {
        continue
    }

    $jsonDocumentCount++
    try {
        $jsonNode = Get-Content -LiteralPath $jsonFile.FullName -Raw -Encoding utf8 |
            ConvertFrom-Json -Depth 100
        Visit-JsonNode -Node $jsonNode -JsonRelativePath $jsonRelativePath
    }
    catch {
        $failures.Add("Governed documentation JSON is malformed: $($jsonRelativePath): $($_.Exception.Message)")
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
Write-Host (
    "Documentation index check passed for {0} top-level Markdown file(s), {1} research domain(s), {2} authority-surface local link(s), and {3}/{4} governed path field(s) across {5} JSON document(s)." -f
    ((Get-ChildItem -LiteralPath $docsRootPath -File -Filter *.md).Count - 1),
    $researchDomainCount,
    $validatedLocalLinkCount,
    $validatedJsonPathCount,
    $jsonPathFieldCount,
    $jsonDocumentCount
)
