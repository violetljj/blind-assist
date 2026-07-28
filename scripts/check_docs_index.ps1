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
foreach ($match in [regex]::Matches($indexText, $linkPattern)) {
    $target = $match.Groups[1].Value
    if ($target -match '^[a-zA-Z][a-zA-Z0-9+.-]*:' -or $target.StartsWith('/')) {
        continue
    }

    $targetPath = Join-Path $docsRootPath $target
    if (-not (Test-Path -LiteralPath $targetPath)) {
        $failures.Add("Documentation index link target is missing: $target")
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
