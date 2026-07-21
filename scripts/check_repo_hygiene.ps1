param(
    [string]$BaseRef = $env:BASE_REF,
    [switch]$AllTracked
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

foreach ($path in $paths) {
    # Deleting a historical forbidden artifact is cleanup, not a new violation.
    if (Test-DeletedOnly $path) {
        continue
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

Write-Host "Repository hygiene check passed for $($paths.Count) changed path(s)."
