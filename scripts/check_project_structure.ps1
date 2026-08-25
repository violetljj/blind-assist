[CmdletBinding()]
param(
    [string]$RepoRoot = '',
    [string]$PolicyPath = (Join-Path $PSScriptRoot 'policy/project_structure.json'),
    [string]$BaseRef = $env:BASE_REF,
    [datetime]$AsOfDate = (Get-Date).Date
)

$ErrorActionPreference = 'Stop'

function Normalize-RepoPath([string]$Path) {
    $normalized = $Path.Replace('\', '/')
    while ($normalized.StartsWith('./', [StringComparison]::Ordinal)) {
        $normalized = $normalized.Substring(2)
    }
    return $normalized
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

# Top-level tracked directories are a reviewed architecture boundary. Local ignored
# toolchains, caches and artifact junctions are intentionally outside this check.
$rootDirectories = @(
    $repoFiles |
        Where-Object { $_ -match '/' } |
        ForEach-Object { $_.Split('/', 2)[0] } |
        Sort-Object -Unique
)
$allowedRootDirectories = @(
    $policy.root_directory_allowlist |
        ForEach-Object { ([string]$_).Replace('\', '/').TrimEnd('/') } |
        Where-Object { $_ } |
        Sort-Object -Unique
)
if ($allowedRootDirectories.Count -eq 0) {
    $failures.Add('Project structure policy must declare root_directory_allowlist.')
}
foreach ($extra in @($rootDirectories | Where-Object { $allowedRootDirectories -notcontains $_ })) {
    $failures.Add("Unreviewed repository root directory: $extra/. Use an existing responsibility layer or update the reviewed root directory allowlist.")
}
foreach ($missing in @($allowedRootDirectories | Where-Object { $rootDirectories -notcontains $_ })) {
    $failures.Add("Root directory allowlist is stale; missing tracked directory: $missing/")
}

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
$moduleIndexPath = Resolve-FromRepo "$researchRoot/MODULE_INDEX.md"
$moduleIndexText = ''
if (-not (Test-Path -LiteralPath $moduleIndexPath -PathType Leaf)) {
    $failures.Add("Research Module index is missing: $researchRoot/MODULE_INDEX.md")
}
else {
    $moduleIndexText = Read-Utf8Text $moduleIndexPath
}
$moduleFamiliesPath = Resolve-FromRepo "$researchRoot/module_families.json"
$moduleFamilies = $null
if (-not (Test-Path -LiteralPath $moduleFamiliesPath -PathType Leaf)) {
    $failures.Add("Research Module family manifest is missing: $researchRoot/module_families.json")
}
else {
    try {
        $moduleFamilies = Get-Content -LiteralPath $moduleFamiliesPath -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($family in @($moduleFamilies.family_order)) {
            if (-not ($moduleFamilies.families.PSObject.Properties.Name -contains [string]$family)) {
                $failures.Add("Research Module family manifest lists an undefined family: $family")
            }
        }
    }
    catch {
        $failures.Add("Research Module family manifest is invalid: $($_.Exception.Message)")
        $moduleFamilies = $null
    }
}
$archiveModuleNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$archiveManifestRelative = Normalize-RepoPath ([string]$policy.research_archive_manifest)
if (-not [string]::IsNullOrWhiteSpace($archiveManifestRelative)) {
    $archiveManifestPath = Resolve-FromRepo $archiveManifestRelative
    if (-not (Test-Path -LiteralPath $archiveManifestPath -PathType Leaf)) {
        $failures.Add("Research archive manifest is missing: $archiveManifestRelative")
    }
    else {
        try {
            $archiveManifest = Get-Content -LiteralPath $archiveManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ([string]$archiveManifest.contract.status -ne 'archive' -or
                [bool]$archiveManifest.contract.current_execution -or
                -not [bool]$archiveManifest.contract.path_stable) {
                $failures.Add('Research archive manifest contract must be archive, non-executable, and path-stable.')
            }
            foreach ($archiveModule in @($archiveManifest.modules)) {
                $archiveModuleName = [string]$archiveModule
                if ([string]::IsNullOrWhiteSpace($archiveModuleName) -or $archiveModuleName -notmatch '^[a-z0-9_]+$') {
                    $failures.Add("Research archive manifest has an invalid module name: $archiveModuleName")
                    continue
                }
                if (-not $archiveModuleNames.Add($archiveModuleName)) {
                    $failures.Add("Research archive manifest duplicates module: $archiveModuleName")
                }
            }
        }
        catch {
            $failures.Add("Research archive manifest is invalid: $($_.Exception.Message)")
        }
    }
}
$hftfRegistry = Resolve-FromRepo "$researchRoot/hftf/INDEX.md"
if (-not (Test-Path -LiteralPath $hftfRegistry -PathType Leaf)) {
    $failures.Add("HFTF role index is missing: $researchRoot/hftf/INDEX.md")
}
$hftfRolesPath = Resolve-FromRepo "$researchRoot/hftf/roles.json"
if (-not (Test-Path -LiteralPath $hftfRolesPath -PathType Leaf)) {
    $failures.Add("HFTF role manifest is missing: $researchRoot/hftf/roles.json")
}
else {
    try {
        $hftfRoles = Get-Content -LiteralPath $hftfRolesPath -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($role in @($hftfRoles.role_order)) {
            if (-not ($hftfRoles.roles.PSObject.Properties.Name -contains [string]$role)) {
                $failures.Add("HFTF role manifest lists an undefined role: $role")
            }
        }
        $hftfPrefix = "$researchRoot/hftf/"
        $hftfSupportCount = 0
        foreach ($file in @($repoFiles | Where-Object { $_.StartsWith($hftfPrefix, [StringComparison]::OrdinalIgnoreCase) })) {
            $relative = $file.Substring($hftfPrefix.Length)
            $matchedRole = $null
            foreach ($role in @($hftfRoles.role_order)) {
                $patterns = @($hftfRoles.roles.$role.patterns)
                foreach ($pattern in $patterns) {
                    if ($relative -match [string]$pattern) {
                        $matchedRole = [string]$role
                        break
                    }
                }
                if ($null -ne $matchedRole) { break }
            }
            if ($null -eq $matchedRole) {
                $failures.Add("HFTF file has no role manifest match: $relative")
            }
            elseif ($matchedRole -eq 'support') {
                $hftfSupportCount++
            }
        }
        $hftfSupportMaxFiles = [int]$policy.hftf_support_max_files
        if ($hftfSupportCount -gt $hftfSupportMaxFiles) {
            $failures.Add("HFTF support role has $hftfSupportCount Git-visible file(s); policy maximum is $hftfSupportMaxFiles. Classify them before merging.")
        }
    }
    catch {
        $failures.Add("HFTF role manifest is invalid: $($_.Exception.Message)")
    }
}
$allResearchDirectoryNames = @(
    $repoFiles |
        Where-Object { $_.StartsWith($researchPrefix, [StringComparison]::OrdinalIgnoreCase) } |
        ForEach-Object {
            $remainder = $_.Substring($researchPrefix.Length)
            if ($remainder -match '^([^/]+)/') { $Matches[1] }
        } |
        Where-Object { $_ } |
        Sort-Object -Unique
)
$moduleNames = @($allResearchDirectoryNames | Where-Object { -not $archiveModuleNames.Contains($_) })
foreach ($archiveModuleName in $archiveModuleNames) {
    if ($allResearchDirectoryNames -notcontains $archiveModuleName) {
        $failures.Add("Research archive manifest points to a missing package: $archiveModuleName")
    }
    if ($null -ne $moduleFamilies) {
        $matchedArchiveFamilies = @(
            foreach ($family in @($moduleFamilies.family_order)) {
                foreach ($pattern in @($moduleFamilies.families.$family.patterns)) {
                    if ($archiveModuleName -match [string]$pattern) {
                        [string]$family
                        break
                    }
                }
            }
        )
        if ($matchedArchiveFamilies.Count -gt 0) {
            $failures.Add("Archived research package must not match a current family: $archiveModuleName [$($matchedArchiveFamilies -join ', ')]")
        }
    }
}
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
    if (-not [string]::IsNullOrWhiteSpace($moduleIndexText) -and -not $moduleIndexText.Contains("($moduleName/README.md)")) {
        $failures.Add("Research Module is missing from MODULE_INDEX.md: $moduleName")
    }
    if ($null -ne $moduleFamilies) {
        $matchedFamilies = @()
        foreach ($family in @($moduleFamilies.family_order)) {
            foreach ($pattern in @($moduleFamilies.families.$family.patterns)) {
                if ($moduleName -match [string]$pattern) {
                    $matchedFamilies += [string]$family
                    break
                }
            }
        }
        if ($matchedFamilies.Count -ne 1) {
            $failures.Add("Research Module must match exactly one family: $moduleName matched $($matchedFamilies.Count) [$($matchedFamilies -join ', ')]")
        }
        else {
            $dynamicTruth = Normalize-RepoPath ([string]$moduleFamilies.families.($matchedFamilies[0]).dynamic_truth)
            if ([string]::IsNullOrWhiteSpace($dynamicTruth) -or $repoFiles -notcontains $dynamicTruth) {
                $failures.Add("Research Module family dynamic truth is missing for ${moduleName}: $dynamicTruth")
            }
        }
    }
}

# Current navigation may summarize a route, but it must not silently become a
# second, divergent execution authority. The policy keeps machine-checkable
# ownership markers and validates category summaries against current route
# READMEs while leaving dated closure documents immutable.
$currentTruthPolicy = $policy.current_truth
if ($null -ne $currentTruthPolicy) {
    $currentTruthPaths = @(
        [string]$currentTruthPolicy.root_readme_path,
        [string]$currentTruthPolicy.algorithm_current_path,
        [string]$currentTruthPolicy.system_current_path,
        [string]$currentTruthPolicy.scripts_index_path,
        [string]$currentTruthPolicy.research_registry_path,
        [string]$currentTruthPolicy.module_count_owner_path
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    foreach ($path in $currentTruthPaths) {
        if (-not (Test-Path -LiteralPath (Resolve-FromRepo $path) -PathType Leaf)) {
            $failures.Add("Current-truth governance path is missing: $path")
        }
    }

    $rootReadmePath = Resolve-FromRepo ([string]$currentTruthPolicy.root_readme_path)
    if (Test-Path -LiteralPath $rootReadmePath -PathType Leaf) {
        $rootReadmeText = Read-Utf8Text $rootReadmePath
        $ownerMarker = [string]$currentTruthPolicy.root_research_owner_marker
        if ([string]::IsNullOrWhiteSpace($ownerMarker) -or -not $rootReadmeText.Contains($ownerMarker)) {
            $failures.Add("Root README must delegate dynamic research status with marker: $ownerMarker")
        }
        $rootStatusMatch = [regex]::Match(
            $rootReadmeText,
            '(?ms)^##\s+当前状态\s*\r?\n(?<body>.*?)(?=^##\s+|\z)'
        )
        if (-not $rootStatusMatch.Success) {
            $failures.Add('Root README is missing the product-owned 当前状态 section.')
        }
        else {
            $rootStatusBody = $rootStatusMatch.Groups['body'].Value
            if ($rootStatusBody -notmatch '\]\(docs/research/README\.md\)') {
                $failures.Add('Root README 当前状态 must link the delegated project research entry: docs/research/README.md')
            }
            foreach ($pattern in @($currentTruthPolicy.root_status_forbidden_patterns | ForEach-Object { [string]$_ })) {
                if ($pattern -and [regex]::IsMatch($rootStatusBody, $pattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
                    $failures.Add("Root README 当前状态 duplicates route-owned dynamic research text: $pattern")
                }
            }
        }
    }

    foreach ($navigationPath in @(
        [string]$currentTruthPolicy.scripts_index_path,
        [string]$currentTruthPolicy.research_registry_path
    )) {
        $absoluteNavigationPath = Resolve-FromRepo $navigationPath
        if (-not (Test-Path -LiteralPath $absoluteNavigationPath -PathType Leaf)) {
            continue
        }
        $navigationText = Read-Utf8Text $absoluteNavigationPath
        if ([regex]::IsMatch($navigationText, '\d+\s*个研究\s*Module', [Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
            $failures.Add("Navigation duplicates the machine-owned research Module count: $navigationPath")
        }
        if ($navigationPath -eq [string]$currentTruthPolicy.scripts_index_path -and $navigationText.Contains('当前论文研究主线')) {
            $failures.Add("Script navigation duplicates route-owned research mainline status: $navigationPath")
        }
    }

    $moduleCountOwnerPath = Resolve-FromRepo ([string]$currentTruthPolicy.module_count_owner_path)
    if (Test-Path -LiteralPath $moduleCountOwnerPath -PathType Leaf) {
        $moduleCountText = Read-Utf8Text $moduleCountOwnerPath
        $moduleCountMatch = [regex]::Match($moduleCountText, '(?im)^状态：.*?(?<listed>\d+)-of-(?<total>\d+)')
        if (-not $moduleCountMatch.Success) {
            $failures.Add("Module count owner lacks an N-of-N status marker: $($currentTruthPolicy.module_count_owner_path)")
        }
        else {
            $listedCount = [int]$moduleCountMatch.Groups['listed'].Value
            $totalCount = [int]$moduleCountMatch.Groups['total'].Value
            if ($listedCount -ne $moduleNames.Count -or $totalCount -ne $moduleNames.Count) {
                $failures.Add("Research Module count drift: index says $listedCount-of-$totalCount, Git-visible structure has $($moduleNames.Count).")
            }
        }
    }

    $systemCurrentPath = Resolve-FromRepo ([string]$currentTruthPolicy.system_current_path)
    if (Test-Path -LiteralPath $systemCurrentPath -PathType Leaf) {
        $systemCurrentLines = [IO.File]::ReadAllLines($systemCurrentPath, [Text.Encoding]::UTF8)
        $deploymentRow = $systemCurrentLines | Where-Object { $_ -match '^\|\s*部署可行性\s*\|' } | Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($deploymentRow)) {
            $failures.Add('System current is missing the deployment-feasibility row.')
        }
        else {
            $deploymentCells = @($deploymentRow.Trim().Trim('|').Split('|') | ForEach-Object { $_.Trim() })
            if ($deploymentCells.Count -ne 5) {
                $failures.Add("System current deployment row must have 5 columns; found $($deploymentCells.Count).")
            }
            else {
                $stateMarker = [string]$currentTruthPolicy.system_route_state_marker
                $successorMarker = [string]$currentTruthPolicy.system_route_successor_marker
                if ($deploymentCells[2].Trim([char]0x60) -ne $stateMarker) {
                    $failures.Add("System current must delegate DepthART state with marker: $stateMarker")
                }
                if (-not $deploymentCells[4].Contains($successorMarker)) {
                    $failures.Add("System current must delegate DepthART successor with marker: $successorMarker")
                }
                if ($deploymentCells[3] -notmatch '\]\(hftf/README\.md\)') {
                    $failures.Add('System current deployment row must point to the DepthART route current: hftf/README.md')
                }
            }
        }
    }

    $algorithmCurrentPath = Resolve-FromRepo ([string]$currentTruthPolicy.algorithm_current_path)
    if (Test-Path -LiteralPath $algorithmCurrentPath -PathType Leaf) {
        $algorithmCurrentDirectory = Split-Path -Parent $algorithmCurrentPath
        $algorithmCurrentLines = [IO.File]::ReadAllLines($algorithmCurrentPath, [Text.Encoding]::UTF8)
        $routeNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        $routeCount = 0
        foreach ($line in $algorithmCurrentLines) {
            if ($line -notmatch '^\|' -or $line -match '^\|\s*---' -or $line -match '^\|\s*路线\s*\|') {
                continue
            }
            $cells = @($line.Trim().Trim('|').Split('|') | ForEach-Object { $_.Trim() })
            if ($cells.Count -ne 7) {
                $failures.Add("Algorithm current route row must have 7 columns; found $($cells.Count): $line")
                continue
            }
            $routeCount++
            $routeName = $cells[0]
            if ([string]::IsNullOrWhiteSpace($routeName) -or -not $routeNames.Add($routeName)) {
                $failures.Add("Algorithm current route name is empty or duplicated: $routeName")
            }
            foreach ($cellIndex in @(1, 2, 3, 4, 5, 6)) {
                if ([string]::IsNullOrWhiteSpace($cells[$cellIndex])) {
                    $failures.Add("Algorithm current route '$routeName' has an empty required column: $cellIndex")
                }
            }
            if ($cells[6] -notin @('是', '否')) {
                $failures.Add("Algorithm current route '$routeName' has invalid default-App impact: $($cells[6])")
            }

            $truthLink = [regex]::Match($cells[3], '\]\((?<target>[^)#]+)(?:#[^)]*)?\)')
            if (-not $truthLink.Success) {
                $failures.Add("Algorithm current route '$routeName' lacks a local unique-truth link.")
                continue
            }
            $truthTarget = $truthLink.Groups['target'].Value.Trim('<', '>')
            if ($truthTarget -match '^[a-zA-Z][a-zA-Z0-9+.-]*:' -or [IO.Path]::IsPathRooted($truthTarget)) {
                $failures.Add("Algorithm current route '$routeName' unique truth must be repository-local: $truthTarget")
                continue
            }
            $truthPath = [IO.Path]::GetFullPath((Join-Path $algorithmCurrentDirectory $truthTarget))
            $truthRelative = Normalize-RepoPath ([IO.Path]::GetRelativePath($script:ResolvedRepoRoot, $truthPath))
            if ($truthRelative.StartsWith('../', [StringComparison]::Ordinal) -or -not (Test-Path -LiteralPath $truthPath -PathType Leaf)) {
                $failures.Add("Algorithm current route '$routeName' unique truth is missing or outside the repository: $truthTarget")
                continue
            }

            # Only a current route README is expected to repeat its category summary.
            # Dated closure documents remain immutable and are intentionally skipped.
            if ([IO.Path]::GetFileName($truthPath) -eq 'README.md') {
                $truthText = Read-Utf8Text $truthPath
                $routeStatusMatch = [regex]::Match(
                    $truthText,
                    '(?im)^状态：\s*`(?<status>[^`\r\n]+)`\s*$'
                )
                if (-not $routeStatusMatch.Success) {
                    $failures.Add("Algorithm current route '$routeName' truth lacks one machine-readable current status line.")
                    continue
                }
                $routeStatusTokens = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
                foreach ($routeStatusToken in @(
                    $routeStatusMatch.Groups['status'].Value -split '\s*/\s*' |
                        ForEach-Object { $_.Trim() } |
                        Where-Object { $_ }
                )) {
                    [void]$routeStatusTokens.Add($routeStatusToken)
                }
                if (-not $routeStatusTokens.Contains('current')) {
                    $failures.Add("Algorithm current route '$routeName' truth status line is not marked current.")
                }

                $defaultAppMarker = if ($cells[6] -eq '否') {
                    [string]$currentTruthPolicy.default_app_unchanged_marker
                }
                else {
                    [string]$currentTruthPolicy.default_app_changed_marker
                }
                if ([string]::IsNullOrWhiteSpace($defaultAppMarker) -or -not $routeStatusTokens.Contains($defaultAppMarker)) {
                    $failures.Add("Algorithm current route '$routeName' default-App impact '$($cells[6])' is not synchronized by route marker: $defaultAppMarker")
                }
            }
        }
        if ($routeCount -eq 0) {
            $failures.Add('Algorithm current contains no governed route rows.')
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

Write-Host "Project structure check passed: $($rootDirectories.Count) root directories, $($rootFiles.Count) root scripts, $($moduleNames.Count) research Module(s), log within budget."
exit 0
