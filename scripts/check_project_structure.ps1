[CmdletBinding()]
param(
    [string]$RepoRoot = '',
    [string]$PolicyPath = (Join-Path $PSScriptRoot 'policy/project_structure.json'),
    [string]$BaseRef = $env:BASE_REF,
    [datetime]$AsOfDate = (Get-Date).Date
)

$ErrorActionPreference = 'Stop'

function Normalize-RepoPath([string]$Path) {
    return $Path.Replace('\', '/').TrimStart('./')
}

function Resolve-FromRepo([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $script:ResolvedRepoRoot $Path))
}

function Read-Utf8Text([string]$Path) {
    return [IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8)
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $detectedRoot = (& git rev-parse --show-toplevel 2>$null | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($detectedRoot)) {
        throw 'Project structure check must run inside a Git repository or receive -RepoRoot.'
    }
    $RepoRoot = $detectedRoot.Trim()
}

$script:ResolvedRepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$resolvedPolicyPath = Resolve-FromRepo $PolicyPath
if (-not (Test-Path -LiteralPath $resolvedPolicyPath -PathType Leaf)) {
    throw "Project structure policy is missing: $resolvedPolicyPath"
}
$policy = Read-Utf8Text $resolvedPolicyPath | ConvertFrom-Json
$failures = [Collections.Generic.List[string]]::new()

$repoFiles = @(& git -C $script:ResolvedRepoRoot ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to enumerate repository files: $script:ResolvedRepoRoot"
}
$repoFiles = @(
    $repoFiles |
        ForEach-Object { Normalize-RepoPath $_ } |
        Where-Object { $_ -and (Test-Path -LiteralPath (Resolve-FromRepo $_) -PathType Leaf) } |
        Sort-Object -Unique
)

# Root scripts are an explicit Interface. Any addition or removal requires a reviewed policy change.
$scriptsRoot = (Normalize-RepoPath ([string]$policy.scripts_root)).TrimEnd('/')
$rootPrefix = "$scriptsRoot/"
$rootFiles = @(
    $repoFiles |
        Where-Object { $_.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase) } |
        ForEach-Object { $_.Substring($rootPrefix.Length) } |
        Where-Object { $_ -notmatch '/' } |
        Sort-Object -Unique
)
$allowlistPath = Resolve-FromRepo ([string]$policy.root_allowlist_path)
if (-not (Test-Path -LiteralPath $allowlistPath -PathType Leaf)) {
    $failures.Add("Root script allowlist is missing: $($policy.root_allowlist_path)")
    $allowedRootFiles = @()
}
else {
    $allowedRootFiles = @(
        Get-Content -LiteralPath $allowlistPath -Encoding UTF8 |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -and -not $_.StartsWith('#') } |
            Sort-Object -Unique
    )
}
foreach ($extra in @($rootFiles | Where-Object { $allowedRootFiles -notcontains $_ })) {
    $failures.Add("Unreviewed scripts root file: $rootPrefix$extra. Put research Implementation under scripts/research/<domain>/ or update the reviewed allowlist for a stable Interface.")
}
foreach ($missing in @($allowedRootFiles | Where-Object { $rootFiles -notcontains $_ })) {
    $failures.Add("Root script allowlist is stale; missing file: $rootPrefix$missing")
}

# A newly added stable root Interface must also be discoverable from the stable script index.
# Do not retroactively judge files that predate this gate: if the selected base did not yet
# contain the policy, the branch is bootstrapping the gate and has no reviewed comparison point.
if ($BaseRef -and $BaseRef -notmatch '^0+$') {
    $resolvedBase = (& git -C $script:ResolvedRepoRoot rev-parse --verify "$BaseRef^{commit}" 2>$null | Select-Object -First 1)
    if (-not [string]::IsNullOrWhiteSpace($resolvedBase)) {
        $policyRelativePath = Normalize-RepoPath ([IO.Path]::GetRelativePath($script:ResolvedRepoRoot, $resolvedPolicyPath))
        $baseHasPolicy = $false
        if (-not $policyRelativePath.StartsWith('../', [StringComparison]::Ordinal)) {
            & git -C $script:ResolvedRepoRoot cat-file -e "$($resolvedBase.Trim()):$policyRelativePath" 2>$null
            $baseHasPolicy = $LASTEXITCODE -eq 0
        }
        if ($baseHasPolicy) {
            $baseFiles = @(& git -C $script:ResolvedRepoRoot ls-tree -r --name-only $resolvedBase.Trim() -- $scriptsRoot)
            $baseRootFiles = @(
                $baseFiles |
                    ForEach-Object { Normalize-RepoPath $_ } |
                    Where-Object { $_.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase) } |
                    ForEach-Object { $_.Substring($rootPrefix.Length) } |
                    Where-Object { $_ -notmatch '/' } |
                    Sort-Object -Unique
            )
            $scriptsIndexPath = Resolve-FromRepo "$scriptsRoot/README.md"
            $scriptsIndexText = if (Test-Path -LiteralPath $scriptsIndexPath -PathType Leaf) { Read-Utf8Text $scriptsIndexPath } else { '' }
            $indexExemptPatterns = @($policy.root_index_exempt_patterns | ForEach-Object { [string]$_ })
            foreach ($addedRootFile in @($rootFiles | Where-Object { $baseRootFiles -notcontains $_ })) {
                $isIndexExempt = $false
                foreach ($pattern in $indexExemptPatterns) {
                    if ($pattern -and $addedRootFile -match $pattern) {
                        $isIndexExempt = $true
                        break
                    }
                }
                if (-not $isIndexExempt -and -not $scriptsIndexText.Contains($addedRootFile)) {
                    $failures.Add("New root Interface is not indexed in scripts/README.md: $rootPrefix$addedRootFile")
                }
            }
        }
    }
}

# The recent log has a hard budget so history cannot silently become the navigation Interface.
$logPolicy = $policy.development_log
$hardLogLimits = @{ max_lines = 6000; max_bytes = 1200000; max_age_days = 28 }
foreach ($limitName in $hardLogLimits.Keys) {
    if ([long]$logPolicy.$limitName -gt [long]$hardLogLimits[$limitName]) {
        $failures.Add("Development log policy $limitName=$($logPolicy.$limitName) exceeds hard maximum $($hardLogLimits[$limitName]). Archive or shorten content instead of raising the budget.")
    }
}
$logPath = Resolve-FromRepo ([string]$logPolicy.path)
if (-not (Test-Path -LiteralPath $logPath -PathType Leaf)) {
    $failures.Add("Development log is missing: $($logPolicy.path)")
}
else {
    $logLines = [IO.File]::ReadAllLines($logPath, [Text.Encoding]::UTF8)
    $logBytes = (Get-Item -LiteralPath $logPath).Length
    if ($logLines.Count -gt [int]$logPolicy.max_lines) {
        $failures.Add("Development log has $($logLines.Count) lines; maximum is $($logPolicy.max_lines). Archive old month blocks under docs/history/development-log/ without rewriting them.")
    }
    if ($logBytes -gt [long]$logPolicy.max_bytes) {
        $failures.Add("Development log has $logBytes bytes; maximum is $($logPolicy.max_bytes). Link detailed evidence instead of appending raw experiment history.")
    }

    $logText = $logLines -join "`n"
    $dates = [Collections.Generic.List[datetime]]::new()
    $datePatterns = @(
        '(?m)^#{1,3}\s+(?<date>\d{4}-\d{2}-\d{2})(?:\s|$|[：—-])',
        '(?m)^-\s*(?:时间|Time)[:：]\s*(?<date>\d{4}-\d{2}-\d{2})'
    )
    foreach ($pattern in $datePatterns) {
        foreach ($match in [regex]::Matches($logText, $pattern)) {
            $parsed = [datetime]::MinValue
            if ([datetime]::TryParseExact(
                $match.Groups['date'].Value,
                'yyyy-MM-dd',
                [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::None,
                [ref]$parsed
            )) {
                $dates.Add($parsed.Date)
            }
        }
    }
    if ($dates.Count -eq 0) {
        $failures.Add('Development log has no parseable dated entry.')
    }
    else {
        $oldest = $dates | Sort-Object | Select-Object -First 1
        $cutoff = $AsOfDate.Date.AddDays(-[int]$logPolicy.max_age_days)
        if ($oldest -lt $cutoff) {
            $failures.Add("Development log oldest entry is $($oldest.ToString('yyyy-MM-dd')); retention cutoff is $($cutoff.ToString('yyyy-MM-dd')). Archive complete old month blocks and retain links in the root log.")
        }
    }
}

# Every research Module owns a small, testable contract at its directory Interface.
$researchRoot = (Normalize-RepoPath ([string]$policy.research_root)).TrimEnd('/')
$researchPrefix = "$researchRoot/"
$researchRegistry = Resolve-FromRepo "$researchRoot/REGISTRY.md"
if (-not (Test-Path -LiteralPath $researchRegistry -PathType Leaf)) {
    $failures.Add("Research registry is missing: $researchRoot/REGISTRY.md")
}
$hftfRegistry = Resolve-FromRepo "$researchRoot/hftf/INDEX.md"
if (-not (Test-Path -LiteralPath $hftfRegistry -PathType Leaf)) {
    $failures.Add("HFTF role index is missing: $researchRoot/hftf/INDEX.md")
}
$moduleNames = @(
    $repoFiles |
        Where-Object { $_.StartsWith($researchPrefix, [StringComparison]::OrdinalIgnoreCase) } |
        ForEach-Object {
            $remainder = $_.Substring($researchPrefix.Length)
            if ($remainder -match '^([^/]+)/') { $Matches[1] }
        } |
        Where-Object { $_ } |
        Sort-Object -Unique
)
foreach ($moduleName in $moduleNames) {
    $readmeRelative = "$researchPrefix$moduleName/README.md"
    $readmePath = Resolve-FromRepo $readmeRelative
    if (-not (Test-Path -LiteralPath $readmePath -PathType Leaf)) {
        $failures.Add("Research Module lacks README contract: $readmeRelative")
        continue
    }
    $readmeText = Read-Utf8Text $readmePath
    foreach ($marker in @($policy.research_readme_required_markers)) {
        if (-not $readmeText.Contains([string]$marker)) {
            $failures.Add("Research Module $moduleName README lacks required marker: $marker")
        }
    }
}

# Callers may use stable Adapters, but must not learn a campaign's internal file layout.
$referenceSourceAllowlist = @($policy.internal_reference_source_allowlist | ForEach-Object { Normalize-RepoPath ([string]$_) })
$immutableReferenceExceptionPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$declaredImmutableReferencePaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($exception in @($policy.immutable_internal_reference_exceptions)) {
    $rawPath = ([string]$exception.path).Replace('\', '/').Trim()
    $sha256 = ([string]$exception.sha256).Trim().ToUpperInvariant()
    $reason = ([string]$exception.reason).Trim()

    if ([string]::IsNullOrWhiteSpace($rawPath)) {
        $failures.Add('Immutable internal-reference exception has an empty path.')
        continue
    }
    if ([IO.Path]::IsPathRooted($rawPath) -or $rawPath -match '(^|/)\.\.(/|$)') {
        $failures.Add("Immutable internal-reference exception path must stay repository-relative: $rawPath")
        continue
    }

    $path = Normalize-RepoPath $rawPath
    if (-not $declaredImmutableReferencePaths.Add($path)) {
        $failures.Add("Duplicate immutable internal-reference exception path: $path")
        continue
    }
    if ([string]::IsNullOrWhiteSpace($reason)) {
        $failures.Add("Immutable internal-reference exception lacks a reason: $path")
        continue
    }
    if ($sha256 -notmatch '^[0-9A-F]{64}$') {
        $failures.Add("Immutable internal-reference exception has invalid SHA256: $path")
        continue
    }
    if ($repoFiles -notcontains $path) {
        $failures.Add("Immutable internal-reference exception path is missing from the repository: $path")
        continue
    }

    $absolute = Resolve-FromRepo $path
    $actualSha256 = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actualSha256 -ne $sha256) {
        $failures.Add("Immutable internal-reference exception hash drift in ${path}: expected $sha256, actual $actualSha256")
        continue
    }
    [void]$immutableReferenceExceptionPaths.Add($path)
}
$textExtensions = @('.json', '.kt', '.kts', '.md', '.ps1', '.py', '.txt', '.yaml', '.yml')
$internalPathPattern = 'scripts[\\/]+research[\\/]+[A-Za-z0-9_.-]+[\\/]+[A-Za-z0-9_.-]+\.(?:py|ps1)'
foreach ($path in $repoFiles) {
    if (
        $path.StartsWith($researchPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        $referenceSourceAllowlist -contains $path -or
        $immutableReferenceExceptionPaths.Contains($path) -or
        @($policy.historical_reference_skip_prefixes | ForEach-Object { Normalize-RepoPath ([string]$_) } |
            Where-Object { $path.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0
    ) {
        continue
    }
    if ($textExtensions -notcontains [IO.Path]::GetExtension($path).ToLowerInvariant()) {
        continue
    }
    $absolute = Resolve-FromRepo $path
    if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
        continue
    }
    $content = Read-Utf8Text $absolute
    $reference = [regex]::Match($content, $internalPathPattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($reference.Success) {
        $failures.Add("Caller depends on research Implementation path in ${path}: $($reference.Value). Use a stable root Adapter.")
    }
}

# `research.common` is the only cross-domain shared Module. Domain-private imports fail closed.
$researchImportPattern = '(?m)^\s*(?:from|import)\s+research\.([A-Za-z0-9_]+)'
foreach ($path in @($repoFiles | Where-Object { $_ -like 'scripts/*.py' -or $_ -like 'scripts/research/*.py' -or $_ -like 'scripts/research/*/*.py' })) {
    $absolute = Resolve-FromRepo $path
    if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
        continue
    }
    $sourceDomain = ''
    if ($path -match '^scripts/research/([^/]+)/') {
        $sourceDomain = $Matches[1]
    }
    $content = Read-Utf8Text $absolute
    foreach ($match in [regex]::Matches($content, $researchImportPattern)) {
        $targetDomain = $match.Groups[1].Value
        if ($targetDomain -eq 'common' -or $targetDomain -eq $sourceDomain) {
            continue
        }
        $failures.Add("Cross-Module private import in ${path}: research.$targetDomain. Move shared Implementation to research.common or add a stable Adapter.")
    }
}

# Acquisition, annotation, review, admission, acceptance, and release gates are autonomous.
# Reintroducing a human-required authority into an authoritative Interface must fail in CI.
$aiReviewPolicyPath = Resolve-FromRepo 'scripts/policy/ai_review_authority.json'
if (-not (Test-Path -LiteralPath $aiReviewPolicyPath -PathType Leaf)) {
    $failures.Add('Autonomous workflow authority policy is missing: scripts/policy/ai_review_authority.json')
}
else {
    $aiReviewPolicy = Read-Utf8Text $aiReviewPolicyPath | ConvertFrom-Json
    $authorityScanPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $scanExtensions = @($aiReviewPolicy.scan_extensions | ForEach-Object { ([string]$_).ToLowerInvariant() })
    $excludedPrefixes = @($aiReviewPolicy.exclude_path_prefixes | ForEach-Object { Normalize-RepoPath ([string]$_) })
    foreach ($path in @($aiReviewPolicy.scan_paths | ForEach-Object { Normalize-RepoPath ([string]$_) })) {
        $excluded = $false
        foreach ($prefix in $excludedPrefixes) {
            if ($path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                $excluded = $true
                break
            }
        }
        if (-not $excluded) {
            [void]$authorityScanPaths.Add($path)
        }
    }
    foreach ($rootPath in @($aiReviewPolicy.scan_roots | ForEach-Object { Normalize-RepoPath ([string]$_) })) {
        $absoluteRoot = Resolve-FromRepo $rootPath
        if (Test-Path -LiteralPath $absoluteRoot -PathType Leaf) {
            [void]$authorityScanPaths.Add($rootPath)
            continue
        }
        if (-not (Test-Path -LiteralPath $absoluteRoot -PathType Container)) {
            $failures.Add("Autonomous workflow authority scan root is missing: $rootPath")
            continue
        }
        foreach ($file in Get-ChildItem -LiteralPath $absoluteRoot -Recurse -File) {
            if ($scanExtensions.Count -gt 0 -and $scanExtensions -notcontains $file.Extension.ToLowerInvariant()) {
                continue
            }
            $relative = Normalize-RepoPath ([IO.Path]::GetRelativePath($repoRoot, $file.FullName))
            $excluded = $false
            foreach ($prefix in $excludedPrefixes) {
                if ($relative.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                    $excluded = $true
                    break
                }
            }
            if (-not $excluded) {
                [void]$authorityScanPaths.Add($relative)
            }
        }
    }
    foreach ($path in $authorityScanPaths) {
        $absolute = Resolve-FromRepo $path
        if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
            $failures.Add("Autonomous workflow authority scan target is missing: $path")
            continue
        }
        $content = Read-Utf8Text $absolute
        foreach ($pattern in @($aiReviewPolicy.forbidden_patterns | ForEach-Object { [string]$_ })) {
            if ($pattern -and [regex]::IsMatch($content, $pattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
                $failures.Add("Human-required workflow authority reintroduced in ${path}: $pattern")
            }
        }
    }
    foreach ($property in $aiReviewPolicy.required_markers.PSObject.Properties) {
        $path = Normalize-RepoPath ([string]$property.Name)
        $absolute = Resolve-FromRepo $path
        if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
            $failures.Add("Autonomous workflow authority marker target is missing: $path")
            continue
        }
        $content = Read-Utf8Text $absolute
        foreach ($marker in @($property.Value | ForEach-Object { [string]$_ })) {
            if (-not $content.Contains($marker)) {
                $failures.Add("Autonomous workflow authority marker missing in ${path}: $marker")
            }
        }
    }
}

if ($failures.Count -gt 0) {
    Write-Host 'Project structure check failed:'
    foreach ($failure in $failures) {
        Write-Host " - $failure"
    }
    exit 1
}

Write-Host "Project structure check passed: $($rootFiles.Count) root files, $($moduleNames.Count) research Module(s), log within budget."
exit 0
