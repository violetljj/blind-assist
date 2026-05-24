param(
    [string]$BaseRef = $env:BASE_REF,
    [switch]$AllTracked
)

$ErrorActionPreference = "Stop"

function Normalize-PathForGit([string]$Path) {
    return $Path.Replace("\", "/").Trim()
}

function Get-ChangedPaths {
    if ($AllTracked) {
        return git ls-files | ForEach-Object { Normalize-PathForGit $_ }
    }

    if ($BaseRef -and $BaseRef -notmatch "^0+$") {
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

$paths = @(Get-ChangedPaths | Where-Object { $_ } | Sort-Object -Unique)
$docsArchiveChanged = $paths -contains "docs/APK_ARCHIVE.md"
$failures = New-Object System.Collections.Generic.List[string]

foreach ($path in $paths) {
    if ($path -match "^(\\.gradle/|\\.gradle-local/|\\.android-sdk/|\\.jdk/|\\.python311/|\\.venv-|\\.cache/|\\.kotlin/|app/build/|.*/build/)") {
        $failures.Add("Local build/cache path must not be committed: $path")
        continue
    }

    if ($path -match "^test-artifacts([./-]|$)") {
        $failures.Add("New test artifacts must stay in local evidence storage, not Git: $path")
        continue
    }

    if ($path -match "^releases/apk/.*\\.apk$") {
        if (-not $docsArchiveChanged) {
            $failures.Add("New milestone APKs require docs/APK_ARCHIVE.md to be updated: $path")
        }
        continue
    }

    if ($path -match "\\.apk$") {
        $failures.Add("APK files are only allowed under releases/apk with archive documentation: $path")
        continue
    }

    if ($path -match "\\.(aab|zip|pptx|npy|pt|onnx)$") {
        $failures.Add("Large generated/binary artifact must not be added to Git by default: $path")
        continue
    }
}

if ($failures.Count -gt 0) {
    Write-Host "Repository hygiene check failed:"
    foreach ($failure in $failures) {
        Write-Host " - $failure"
    }
    exit 1
}

Write-Host "Repository hygiene check passed for $($paths.Count) changed path(s)."
