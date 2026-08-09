param(
    [string]$IndexScript = (Join-Path $PSScriptRoot 'check_docs_index.ps1')
)

$ErrorActionPreference = 'Stop'
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("blindassist-doc-index-{0}" -f [guid]::NewGuid().ToString('N'))

function Write-Utf8File([string]$Path, [string]$Content) {
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Assert-IndexResult([string]$Name, [string]$IndexText, [string[]]$OtherFiles, [bool]$ShouldPass) {
    $docsRoot = Join-Path $script:testRoot $Name
    New-Item -ItemType Directory -Force -Path $docsRoot | Out-Null
    $IndexText = $IndexText + "`n[project state](PROJECT_STATE.md)"
    Write-Utf8File (Join-Path $docsRoot 'README.md') $IndexText
    Write-Utf8File (Join-Path $docsRoot 'PROJECT_STATE.md') '# fixture'
    foreach ($file in $OtherFiles) {
        Write-Utf8File (Join-Path $docsRoot $file) '# fixture'
    }
    if (@($OtherFiles | Where-Object { $_ -like 'research/*' }).Count -gt 0) {
        Write-Utf8File (Join-Path $docsRoot 'research/README.md') @'
# research

[algorithm](ALGORITHM_RESEARCH_CURRENT.md)
[data](DATA_RESEARCH_CURRENT.md)
[system](SYSTEM_RESEARCH_CURRENT.md)
'@
        foreach ($entry in @(
            'research/ALGORITHM_RESEARCH_CURRENT.md',
            'research/DATA_RESEARCH_CURRENT.md',
            'research/SYSTEM_RESEARCH_CURRENT.md'
        )) {
            Write-Utf8File (Join-Path $docsRoot $entry) '# fixture'
        }
        if ($IndexText -match 'research/domain/README\.md') {
            $IndexText = $IndexText + "`n[research](research/README.md)"
            Write-Utf8File (Join-Path $docsRoot 'README.md') $IndexText
        }
    }

    & $script:IndexScript -DocsRoot $docsRoot
    $passed = $?
    if ($passed -ne $ShouldPass) {
        throw "Scenario '$Name' expected pass=$ShouldPass but got pass=$passed."
    }
    Write-Host "PASS: $Name"
}

function Assert-AuthoritySurfaceLinkResult(
    [string]$Name,
    [string]$RelativePath,
    [string]$Content,
    [bool]$ShouldPass
) {
    $docsRoot = Join-Path $script:testRoot $Name
    New-Item -ItemType Directory -Force -Path $docsRoot | Out-Null
    Write-Utf8File (Join-Path $docsRoot 'README.md') @'
# docs

[project state](PROJECT_STATE.md)
[research](research/README.md)
[domain](research/domain/README.md)
'@
    Write-Utf8File (Join-Path $docsRoot 'PROJECT_STATE.md') '# fixture'
    Write-Utf8File (Join-Path $docsRoot 'research/README.md') @'
# research

[algorithm](ALGORITHM_RESEARCH_CURRENT.md)
[data](DATA_RESEARCH_CURRENT.md)
[system](SYSTEM_RESEARCH_CURRENT.md)
'@
    foreach ($entry in @(
        'research/ALGORITHM_RESEARCH_CURRENT.md',
        'research/DATA_RESEARCH_CURRENT.md',
        'research/SYSTEM_RESEARCH_CURRENT.md'
    )) {
        Write-Utf8File (Join-Path $docsRoot $entry) '# fixture'
    }
    Write-Utf8File (Join-Path $docsRoot 'research/domain/README.md') '# domain'
    Write-Utf8File (Join-Path $docsRoot $RelativePath) $Content

    & $script:IndexScript -DocsRoot $docsRoot
    $passed = $?
    if ($passed -ne $ShouldPass) {
        throw "Scenario '$Name' expected pass=$ShouldPass but got pass=$passed."
    }
    Write-Host "PASS: $Name"
}

try {
    $script:IndexScript = (Resolve-Path -LiteralPath $IndexScript).Path
    New-Item -ItemType Directory -Path $testRoot | Out-Null
    Assert-IndexResult -Name 'indexed-local-link' -IndexText '[current](CURRENT.md)' -OtherFiles @('CURRENT.md') -ShouldPass $true
    Assert-IndexResult -Name 'unindexed-file' -IndexText '# index' -OtherFiles @('ORPHAN.md') -ShouldPass $false
    Assert-IndexResult -Name 'missing-local-link' -IndexText '[missing](MISSING.md)' -OtherFiles @() -ShouldPass $false
    Assert-IndexResult -Name 'indexed-research-domain' -IndexText '[domain](research/domain/README.md)' -OtherFiles @('research/domain/README.md') -ShouldPass $true
    Assert-IndexResult -Name 'research-domain-without-readme' -IndexText '# index' -OtherFiles @('research/domain/SNAPSHOT.md') -ShouldPass $false
    Assert-IndexResult -Name 'research-domain-without-top-index-link' -IndexText '# index' -OtherFiles @('research/domain/README.md') -ShouldPass $false
    Assert-AuthoritySurfaceLinkResult `
        -Name 'current-doc-broken-link' `
        -RelativePath 'research/domain/README.md' `
        -Content "# domain`n`n状态：current`n`n[missing](MISSING.md)`n" `
        -ShouldPass $false
    Assert-AuthoritySurfaceLinkResult `
        -Name 'protocol-broken-link' `
        -RelativePath 'research/domain/FIXTURE_PROTOCOL_2026-08-10.md' `
        -Content "# protocol`n`n[missing](MISSING.md)`n" `
        -ShouldPass $false
    Assert-AuthoritySurfaceLinkResult `
        -Name 'archive-broken-link-is-historical' `
        -RelativePath 'research/domain/archive/FIXTURE_PROTOCOL_2026-08-10.md' `
        -Content "# archived protocol`n`n[missing](MISSING.md)`n" `
        -ShouldPass $true

    $missingCategoryRoot = Join-Path $testRoot 'research-missing-category'
    New-Item -ItemType Directory -Force -Path (Join-Path $missingCategoryRoot 'research') | Out-Null
    Write-Utf8File (Join-Path $missingCategoryRoot 'README.md') '# docs'
    Write-Utf8File (Join-Path $missingCategoryRoot 'PROJECT_STATE.md') '# fixture'
    Write-Utf8File (Join-Path $missingCategoryRoot 'research/README.md') '[algorithm](ALGORITHM_RESEARCH_CURRENT.md)'
    Write-Utf8File (Join-Path $missingCategoryRoot 'research/ALGORITHM_RESEARCH_CURRENT.md') '# algorithm'
    & $script:IndexScript -DocsRoot $missingCategoryRoot
    if ($?) {
        throw "Scenario 'research-missing-category' expected failure but passed."
    }
    Write-Host 'PASS: research-missing-category'
    Write-Host 'Documentation index smoke tests passed.'
}
finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
