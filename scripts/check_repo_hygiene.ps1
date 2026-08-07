param(
    [string]$BaseRef = $env:BASE_REF,
    [switch]$AllTracked,
    [switch]$IncludeStructure,
    [switch]$SkipStructure
)

$ErrorActionPreference = 'Stop'

function Normalize-PathForGit([string]$Path) {
    return $Path.Replace('\', '/').Trim()
}

function Get-ChangedPaths {
    if ($AllTracked) {
        return git ls-files | ForEach-Object { Normalize-PathForGit $_ }
    }

    if ($BaseRef -and $BaseRef -notmatch '^0+$') {
        $resolved = git rev-parse --verify "$BaseRef^{commit}" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $paths = git diff --name-only "$BaseRef...HEAD" | ForEach-Object { Normalize-PathForGit $_ }
            if ($paths) {
                return $paths
            }
        }
    }

    $modified = git diff --name-only | ForEach-Object { Normalize-PathForGit $_ }
    $cached = git diff --name-only --cached | ForEach-Object { Normalize-PathForGit $_ }
    $untracked = git ls-files --others --exclude-standard | ForEach-Object { Normalize-PathForGit $_ }
    return @($modified) + @($cached) + @($untracked)
}

function Resolve-BaseCommit {
    if ($AllTracked -or -not $BaseRef -or $BaseRef -match '^0+$') {
        return $null
    }
    $resolved = git rev-parse --verify "$BaseRef^{commit}" 2>$null
    if ($LASTEXITCODE -eq 0 -and $resolved) {
        return $resolved.Trim()
    }
    return $null
}

function Test-DeletedOnly([string]$Path) {
    return $script:DeletedOnlyPaths -contains $Path
}

function Get-DeletedOnlyPaths {
    if ($script:ResolvedBaseRef) {
        return git diff --name-only --diff-filter=D "$script:ResolvedBaseRef...HEAD" |
            ForEach-Object { Normalize-PathForGit $_ }
    }

    $deleted = @(
        git diff --name-only --diff-filter=D
        git diff --name-only --cached --diff-filter=D
    ) | ForEach-Object { Normalize-PathForGit $_ }
    $notDeleted = @(
        git diff --name-only --diff-filter=ACMRTUXB
        git diff --name-only --cached --diff-filter=ACMRTUXB
        git ls-files --others --exclude-standard
    ) | ForEach-Object { Normalize-PathForGit $_ }
    return $deleted | Where-Object { $_ -and -not ($notDeleted -contains $_) } | Sort-Object -Unique
}

$script:ResolvedBaseRef = Resolve-BaseCommit
$script:DeletedOnlyPaths = @(Get-DeletedOnlyPaths)
$paths = @(Get-ChangedPaths | Where-Object { $_ } | Sort-Object -Unique)
$docsArchiveChanged = $paths -contains 'docs/APK_ARCHIVE.md'
$failures = New-Object System.Collections.Generic.List[string]

$repoRoot = (& git rev-parse --show-toplevel 2>$null | Select-Object -First 1)
if ([string]::IsNullOrWhiteSpace($repoRoot)) {
    throw 'Repository hygiene check must run inside a Git repository.'
}
$repoRoot = (Resolve-Path -LiteralPath $repoRoot.Trim()).Path
$forbiddenRootExtensions = @('.pt', '.onnx', '.npy', '.tflite', '.apk', '.aab', '.zip', '.pptx')
foreach ($rootItem in Get-ChildItem -LiteralPath $repoRoot -Force) {
    if (
        (-not $rootItem.PSIsContainer -and $forbiddenRootExtensions -contains $rootItem.Extension.ToLowerInvariant()) -or
        ($rootItem.PSIsContainer -and $rootItem.Name -like '*_saved_model')
    ) {
        $failures.Add(
            "Generated/model artifact must not live at the repository root, even when ignored: $($rootItem.Name). " +
            'Move local payload to artifacts.local/ and keep committed production assets under their owned module.'
        )
    }
}

$dependencyPolicyPaths = @(
    git ls-files -- '*.gradle' '*.gradle.kts' 'gradle/libs.versions.toml'
    git ls-files --others --exclude-standard -- '*.gradle' '*.gradle.kts' 'gradle/libs.versions.toml'
) | ForEach-Object { Normalize-PathForGit $_ } | Where-Object { $_ } | Sort-Object -Unique
foreach ($dependencyPolicyPath in $dependencyPolicyPaths) {
    if (Test-DeletedOnly $dependencyPolicyPath) {
        continue
    }
    $absoluteDependencyPolicyPath = Join-Path $repoRoot $dependencyPolicyPath
    if (-not (Test-Path -LiteralPath $absoluteDependencyPolicyPath -PathType Leaf)) {
        continue
    }
    $lineNumber = 0
    foreach ($line in Get-Content -LiteralPath $absoluteDependencyPolicyPath -Encoding UTF8) {
        $lineNumber += 1
        $nonReproducibleSelector = $null
        if ($line -match '(?i)\b(latest\.(?:release|integration)|[A-Za-z0-9_.-]+-SNAPSHOT)\b') {
            $nonReproducibleSelector = $Matches[1]
        }
        elseif ($line -match '(?i)\b(?:isChanging|changing)\s*=\s*true\b') {
            $nonReproducibleSelector = $Matches[0]
        }
        elseif ($line -match '[''"][^''"]*:[^''"]*:[^''"]*(\+|\[[^''"]*,[^''"]*[\]\)]|\([^''"]*,[^''"]*[\]\)])[^''"]*[''"]') {
            $nonReproducibleSelector = $Matches[1]
        }
        elseif (
            $dependencyPolicyPath -eq 'gradle/libs.versions.toml' -and
            $line -match '^\s*[A-Za-z0-9_.-]+\s*=\s*[''"]([^''"]*(?:\+|latest\.(?:release|integration)|-SNAPSHOT)|[\[\(][^''"]*,[^''"]*[\]\)])[''"]'
        ) {
            $nonReproducibleSelector = $Matches[1]
        }
        if ($nonReproducibleSelector) {
            $failures.Add(
                "Non-reproducible dependency selector is forbidden: " +
                "$dependencyPolicyPath`:$lineNumber ($nonReproducibleSelector)"
            )
        }
    }
}

foreach ($path in $paths) {
    # Removing a forbidden local/generated artifact is repository cleanup, not a new violation.
    # This must run before every path-family rule so a PR can delete historical binaries,
    # caches, APKs, or credentials without the cleanup itself making CI permanently red.
    if (Test-DeletedOnly $path) {
        continue
    }

    if ($path -match '^\.github/workflows/.*\.(yml|yaml)$') {
        $workflowPath = Join-Path $repoRoot $path
        if (Test-Path -LiteralPath $workflowPath -PathType Leaf) {
            $lineNumber = 0
            foreach ($line in Get-Content -LiteralPath $workflowPath -Encoding UTF8) {
                $lineNumber += 1
                if ($line -notmatch '^\s*(?:-\s*)?uses:\s*([^\s#]+)') {
                    continue
                }
                $action = $Matches[1]
                if ($action.StartsWith('./') -or $action.StartsWith('docker://')) {
                    continue
                }
                if ($action -notmatch '^.+@[0-9a-fA-F]{40}$') {
                    $failures.Add(
                        "External GitHub Action must be pinned to a full commit SHA: " +
                        "$path`:$lineNumber ($action)"
                    )
                }
            }
        }
    }

    if ($path -match '^(\.gradle/|\.gradle-local/|\.android-sdk/|\.android-home/|\.jdk/|\.python311/|\.venv-|\.cache/|\.kotlin/|\.kotlin-home/|work/|app/build/|.*/build/)' -or
        $path -match '(^|/)__pycache__/') {
        $failures.Add("Local build/cache path must not be committed: $path")
        continue
    }

    if ($path -match '^test-artifacts([./-]|$)') {
        $failures.Add("New test artifacts must stay in local evidence storage, not Git: $path")
        continue
    }

    if ($path -match '^releases/apk/.*\.apk$') {
        if (-not $docsArchiveChanged) {
            $failures.Add("New milestone APKs require docs/APK_ARCHIVE.md to be updated: $path")
        }
        continue
    }

    if ($path -match '\.apk$') {
        $failures.Add("APK files are only allowed under releases/apk with archive documentation: $path")
        continue
    }

    if ($path -match '(^|/)keystore\.properties$' -or $path -match '\.(jks|keystore)$') {
        $failures.Add("Signing secrets and keystore files must stay local: $path")
        continue
    }

    if ($path -match '\.(aab|zip|pptx|npy|pt|onnx)$') {
        $failures.Add("Large generated/binary artifact must not be added to Git by default: $path")
        continue
    }
}

if ($failures.Count -gt 0) {
    Write-Host 'Repository hygiene check failed:'
    foreach ($failure in $failures) {
        Write-Host " - $failure"
    }
    exit 1
}

if ($IncludeStructure -and -not $SkipStructure) {
    $structureScript = Join-Path $PSScriptRoot 'check_project_structure.ps1'
    if (-not (Test-Path -LiteralPath $structureScript -PathType Leaf)) {
        Write-Host "Repository hygiene check failed:`n - Project structure gate is missing: $structureScript"
        exit 1
    }
    & $structureScript -BaseRef $BaseRef
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Repository hygiene check passed for $($paths.Count) changed path(s)."
