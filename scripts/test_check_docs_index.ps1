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

function Assert-DocumentationFixtureResult(
    [string]$Name,
    [hashtable]$Files,
    [bool]$ShouldPass
) {
    $repoRoot = Join-Path $script:testRoot $Name
    $docsRoot = Join-Path $repoRoot 'docs'
    New-Item -ItemType Directory -Force -Path $docsRoot | Out-Null
    $baseFiles = @{
        'docs/README.md' = @'
# docs

[project state](PROJECT_STATE.md)
[research](research/README.md)
[domain](research/domain/README.md)
'@
        'docs/PROJECT_STATE.md' = '# fixture'
        'docs/research/README.md' = @'
# research

[algorithm](ALGORITHM_RESEARCH_CURRENT.md)
[data](DATA_RESEARCH_CURRENT.md)
[system](SYSTEM_RESEARCH_CURRENT.md)
'@
        'docs/research/ALGORITHM_RESEARCH_CURRENT.md' = '# fixture'
        'docs/research/DATA_RESEARCH_CURRENT.md' = '# fixture'
        'docs/research/SYSTEM_RESEARCH_CURRENT.md' = '# fixture'
        'docs/research/domain/README.md' = '# domain'
    }
    foreach ($entry in $baseFiles.GetEnumerator()) {
        Write-Utf8File (Join-Path $repoRoot $entry.Key) $entry.Value
    }
    foreach ($entry in $Files.GetEnumerator()) {
        Write-Utf8File (Join-Path $repoRoot $entry.Key) $entry.Value
    }

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
    Assert-DocumentationFixtureResult `
        -Name 'archive-readme-broken-link' `
        -Files @{
            'docs/research/domain/archive/README_FULL_HISTORY_2026-08-13.md' = "# archive`n`n[missing](MISSING.md)`n"
        } `
        -ShouldPass $false
    Assert-DocumentationFixtureResult `
        -Name 'archive-readme-parent-link-valid' `
        -Files @{
            'docs/research/domain/archive/README_FULL_HISTORY_2026-08-13.md' = "# archive`n`n[current](../README.md)`n"
        } `
        -ShouldPass $true
    Assert-DocumentationFixtureResult `
        -Name 'current-snapshot-status-conflict' `
        -Files @{
            'docs/research/domain/DOMAIN_CURRENT_SNAPSHOT_2026-08-13.md' = "# snapshot`n`n状态：current`n"
        } `
        -ShouldPass $false
    Assert-DocumentationFixtureResult `
        -Name 'snapshot-status-valid' `
        -Files @{
            'docs/research/domain/DOMAIN_CURRENT_SNAPSHOT_2026-08-13.md' = "# snapshot`n`n状态：snapshot / historical`n"
        } `
        -ShouldPass $true
    Assert-DocumentationFixtureResult `
        -Name 'json-repo-path-valid' `
        -Files @{
            'docs/research/domain/result.json' = '{"implementation_path":"scripts/tool.py"}'
            'scripts/tool.py' = '# fixture'
        } `
        -ShouldPass $true
    Assert-DocumentationFixtureResult `
        -Name 'json-repo-path-missing' `
        -Files @{
            'docs/research/domain/result.json' = '{"implementation_path":"scripts/missing.py"}'
        } `
        -ShouldPass $false
    Assert-DocumentationFixtureResult `
        -Name 'json-local-output-path-ignored' `
        -Files @{
            'docs/research/domain/result.json' = '{"artifact_path":"artifacts.local/evidence/missing.json","build_path":"apps/demo/build/missing.txt"}'
        } `
        -ShouldPass $true
    Assert-DocumentationFixtureResult `
        -Name 'json-nonpath-field-ignored' `
        -Files @{
            'docs/research/domain/result.json' = '{"description":"docs/MISSING.md"}'
        } `
        -ShouldPass $true
    Assert-DocumentationFixtureResult `
        -Name 'json-malformed' `
        -Files @{
            'docs/research/domain/result.json' = '{"path":'
        } `
        -ShouldPass $false
    $longRouteBody = (1..181 | ForEach-Object { "line $_" }) -join "`n"
    Assert-DocumentationFixtureResult `
        -Name 'route-readme-budget' `
        -Files @{
            'docs/research/domain/README.md' = $longRouteBody
        } `
        -ShouldPass $false

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
