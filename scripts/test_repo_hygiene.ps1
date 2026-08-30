param(
    [string]$HygieneScript = (Join-Path $PSScriptRoot 'check_repo_hygiene.ps1')
)

$ErrorActionPreference = 'Stop'

function New-TestRepository([string]$Name) {
    $repository = Join-Path $script:TestRoot $Name
    New-Item -ItemType Directory -Path $repository | Out-Null
    & git -C $repository init --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to initialize test repository: $repository"
    }
    & git -C $repository config user.name 'BlindAssist Hygiene Test'
    & git -C $repository config user.email 'hygiene-test@invalid.local'
    return $repository
}

function Add-TestFile(
    [string]$Repository,
    [string]$RelativePath,
    [string]$Content = 'smoke-test'
) {
    $path = Join-Path $Repository $RelativePath
    $parent = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    [System.IO.File]::WriteAllText($path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Invoke-HygieneCheck([string]$Repository, [string]$BaseRef = '') {
    Push-Location $Repository
    try {
        if ($BaseRef) {
            & $script:HygieneScript -SkipStructure -BaseRef $BaseRef
        }
        else {
            & $script:HygieneScript -SkipStructure
        }
        return $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

function Commit-TestRepository([string]$Repository, [string]$Message) {
    & git -C $Repository add --all
    & git -C $Repository commit --quiet -m $Message
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to commit test repository '$Repository': $Message"
    }
    return (& git -C $Repository rev-parse HEAD).Trim()
}

function Assert-DeletedBinaryCleanupPasses {
    $repository = New-TestRepository 'deleted-binary-cleanup'
    Add-TestFile $repository 'codex/skills-snapshot/historical.zip'
    $base = Commit-TestRepository $repository 'add historical binary fixture'
    Remove-Item -LiteralPath (Join-Path $repository 'codex/skills-snapshot/historical.zip')
    Commit-TestRepository $repository 'remove historical binary fixture' | Out-Null

    $exitCode = Invoke-HygieneCheck $repository $base
    if ($exitCode -ne 0) {
        throw "Deleted historical binary cleanup was expected to pass but exited with code $exitCode."
    }
    Write-Host 'PASS: deleted-binary-cleanup'
}

function Assert-AddedBinaryFromBaseFails {
    $repository = New-TestRepository 'added-binary-from-base'
    Add-TestFile $repository 'notes/readme.md'
    $base = Commit-TestRepository $repository 'add clean base fixture'
    Add-TestFile $repository 'snapshot.zip'
    Commit-TestRepository $repository 'add forbidden binary fixture' | Out-Null

    $exitCode = Invoke-HygieneCheck $repository $base
    if ($exitCode -eq 0) {
        throw 'Added binary relative to a base commit was expected to fail.'
    }
    Write-Host 'PASS: added-binary-from-base'
}

function Assert-IgnoredRootModelFails {
    $repository = New-TestRepository 'ignored-root-model'
    [System.IO.File]::WriteAllText(
        (Join-Path $repository '.gitignore'),
        "*.pt`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    Commit-TestRepository $repository 'add ignored model rule' | Out-Null
    Add-TestFile $repository 'yolo11n.pt'

    $exitCode = Invoke-HygieneCheck $repository
    if ($exitCode -eq 0) {
        throw 'Ignored model payload at the repository root was expected to fail.'
    }
    Write-Host 'PASS: ignored-root-model'
}

function Assert-IgnoredRootNativeArtifactFails {
    $repository = New-TestRepository 'ignored-root-native-artifact'
    [System.IO.File]::WriteAllText(
        (Join-Path $repository '.gitignore'),
        "/*.obj`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    Commit-TestRepository $repository 'add ignored native artifact rule' | Out-Null
    Add-TestFile $repository 'converter.obj'

    $exitCode = Invoke-HygieneCheck $repository
    if ($exitCode -eq 0) {
        throw 'Ignored native compiler artifact at the repository root was expected to fail.'
    }
    Write-Host 'PASS: ignored-root-native-artifact'
}

function Assert-UnpinnedGitHubActionFails {
    $repository = New-TestRepository 'unpinned-github-action'
    Add-TestFile `
        -Repository $repository `
        -RelativePath '.github/workflows/ci.yml' `
        -Content "jobs:`n  test:`n    steps:`n      - uses: actions/checkout@v4`n"

    $exitCode = Invoke-HygieneCheck $repository
    if ($exitCode -eq 0) {
        throw 'GitHub Action pinned only to a movable tag was expected to fail.'
    }
    Write-Host 'PASS: unpinned-github-action'
}

function Assert-PinnedGitHubActionPasses {
    $repository = New-TestRepository 'pinned-github-action'
    Add-TestFile `
        -Repository $repository `
        -RelativePath '.github/workflows/ci.yml' `
        -Content (
            "jobs:`n  test:`n    steps:`n" +
            "      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262`n" +
            "      - uses: ./local-action`n"
        )

    $exitCode = Invoke-HygieneCheck $repository
    if ($exitCode -ne 0) {
        throw "Commit-pinned and local GitHub Actions were expected to pass but exited with code $exitCode."
    }
    Write-Host 'PASS: pinned-github-action'
}

function Assert-NonReproducibleDependencyFails(
    [string]$Name,
    [string]$BuildScript
) {
    $repository = New-TestRepository $Name
    Add-TestFile `
        -Repository $repository `
        -RelativePath 'build.gradle.kts' `
        -Content $BuildScript

    $exitCode = Invoke-HygieneCheck $repository
    if ($exitCode -eq 0) {
        throw "Non-reproducible dependency scenario '$Name' was expected to fail."
    }
    Write-Host "PASS: $Name"
}

function Assert-ExactDependencyPasses {
    $repository = New-TestRepository 'exact-dependency'
    Add-TestFile `
        -Repository $repository `
        -RelativePath 'build.gradle.kts' `
        -Content 'dependencies { implementation("com.example:demo:1.2.3") }'

    $exitCode = Invoke-HygieneCheck $repository
    if ($exitCode -ne 0) {
        throw "Exact dependency coordinate was expected to pass but exited with code $exitCode."
    }
    Write-Host 'PASS: exact-dependency'
}

function Assert-ExtraBlankLineAtEofFails {
    $repository = New-TestRepository 'extra-blank-line-at-eof'
    $fileParameters = @{
        Repository = $repository
        RelativePath = 'notes/readme.md'
        Content = "first line" + [Environment]::NewLine * 2
    }
    Add-TestFile @fileParameters
    & git -C $repository add -- 'notes/readme.md'
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to stage blank-at-EOF fixture.'
    }

    $exitCode = Invoke-HygieneCheck $repository
    if ($exitCode -eq 0) {
        throw 'Extra blank line at EOF was expected to fail.'
    }
    Write-Host 'PASS: extra-blank-line-at-eof'
}

function Assert-HygieneResult(
    [string]$Name,
    [bool]$ShouldPass,
    [string[]]$Paths
) {
    $repository = New-TestRepository $Name
    foreach ($path in $Paths) {
        Add-TestFile $repository $path
    }

    $exitCode = Invoke-HygieneCheck $repository
    $passed = $exitCode -eq 0
    if ($passed -ne $ShouldPass) {
        $expectation = if ($ShouldPass) { 'pass' } else { 'fail' }
        throw "Scenario '$Name' was expected to $expectation but exited with code $exitCode."
    }
    Write-Host "PASS: $Name"
}

$resolvedScript = (Resolve-Path -LiteralPath $HygieneScript).Path
$script:HygieneScript = $resolvedScript
$script:TestRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("blindassist-repo-hygiene-{0}" -f [guid]::NewGuid().ToString('N'))

try {
    New-Item -ItemType Directory -Path $script:TestRoot | Out-Null

    Assert-HygieneResult -Name 'markdown-allowed' -ShouldPass $true -Paths @('notes/readme.md')

    $rejectedPaths = @(
        '.android-home/cache.bin',
        '.kotlin-home/cache.bin',
        'scripts/__pycache__/cache.pyc',
        'core/vision/.cxx/Debug/CMakeCache.txt',
        'work/output.txt',
        'app-debug.apk',
        'snapshot.zip',
        'converter.obj',
        'converter.lib',
        'converter.exp',
        'signing/release.jks',
        'signing/release.keystore',
        'config/keystore.properties',
        ('test-artifacts.' + 'local/device/output.json')
    )
    for ($index = 0; $index -lt $rejectedPaths.Count; $index++) {
        Assert-HygieneResult -Name ("rejected-{0}" -f $index) -ShouldPass $false -Paths @($rejectedPaths[$index])
    }

    Assert-HygieneResult `
        -Name 'milestone-apk-without-docs' `
        -ShouldPass $false `
        -Paths @('releases/apk/blindassist-v9.4.0-debug.apk')

    Assert-HygieneResult `
        -Name 'milestone-apk-with-docs' `
        -ShouldPass $true `
        -Paths @('releases/apk/blindassist-v9.4.0-debug.apk', 'docs/APK_ARCHIVE.md')

    Assert-DeletedBinaryCleanupPasses
    Assert-AddedBinaryFromBaseFails
    Assert-IgnoredRootModelFails
    Assert-IgnoredRootNativeArtifactFails
    Assert-UnpinnedGitHubActionFails
    Assert-PinnedGitHubActionPasses
    Assert-NonReproducibleDependencyFails `
        -Name 'dynamic-plus-dependency' `
        -BuildScript 'dependencies { implementation("com.example:demo:1.+") }'
    Assert-NonReproducibleDependencyFails `
        -Name 'latest-release-dependency' `
        -BuildScript 'dependencies { implementation("com.example:demo:latest.release") }'
    Assert-NonReproducibleDependencyFails `
        -Name 'snapshot-dependency' `
        -BuildScript 'dependencies { implementation("com.example:demo:1.2.3-SNAPSHOT") }'
    Assert-NonReproducibleDependencyFails `
        -Name 'version-range-dependency' `
        -BuildScript 'dependencies { implementation("com.example:demo:[1.0,2.0)") }'
    Assert-NonReproducibleDependencyFails `
        -Name 'changing-module-dependency' `
        -BuildScript 'configurations.all { resolutionStrategy.cacheChangingModulesFor(0, "seconds") }; dependencies { implementation("com.example:demo:1.2.3") { isChanging = true } }'
    Assert-ExactDependencyPasses
    Assert-ExtraBlankLineAtEofFails

    Write-Host 'Repository hygiene smoke tests passed.'
}
finally {
    if (Test-Path -LiteralPath $script:TestRoot) {
        Remove-Item -LiteralPath $script:TestRoot -Recurse -Force
    }
}
