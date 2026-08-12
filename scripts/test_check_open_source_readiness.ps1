$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$checkScript = Join-Path $PSScriptRoot 'check_open_source_readiness.ps1'
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("blindassist-open-source-readiness-" + [guid]::NewGuid().ToString('N'))

function Write-FixtureFile([string]$RelativePath, [string]$Content) {
    $path = Join-Path $testRoot $RelativePath
    $parent = Split-Path -Parent $path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    [System.IO.File]::WriteAllText($path, $Content + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

function Assert-Check([bool]$ExpectedPass, [string]$Name) {
    $passed = $true
    try {
        & $checkScript -RepoRoot $testRoot | Out-Null
    }
    catch {
        $passed = $false
    }
    if ($passed -ne $ExpectedPass) {
        throw "$Name expected pass=$ExpectedPass, got pass=$passed"
    }
    Write-Host "PASS: $Name"
}

try {
    $modelPath = 'app/src/main/assets/model.tflite'
    $labelsPath = 'app/src/main/assets/labels.txt'
    Write-FixtureFile $modelPath 'model-bytes'
    Write-FixtureFile $labelsPath 'labels'
    $modelFile = Join-Path $testRoot $modelPath
    $labelsFile = Join-Path $testRoot $labelsPath
    $modelHash = (Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash
    $labelsHash = (Get-FileHash -LiteralPath $labelsFile -Algorithm SHA256).Hash

    Write-FixtureFile 'README.md' 'CONTRIBUTING.md GOVERNANCE.md docs/MODEL_CARD.md SECURITY.md'
    foreach ($file in @(
        'LICENSE', 'CONTRIBUTING.md', 'SECURITY.md', 'CODE_OF_CONDUCT.md',
        'GOVERNANCE.md', 'CITATION.cff', 'docs/OPEN_SOURCE_PUBLIC_VALUE.md',
        '.github/CODEOWNERS', '.github/dependabot.yml',
        '.github/ISSUE_TEMPLATE/bug_report.yml',
        '.github/ISSUE_TEMPLATE/feature_request.yml',
        '.github/pull_request_template.md'
    )) {
        Write-FixtureFile $file 'fixture'
    }
    Write-FixtureFile 'THIRD_PARTY_NOTICES.md' "$modelPath $labelsPath"
    Write-FixtureFile 'docs/MODEL_CARD.md' "$modelPath $modelHash $labelsPath $labelsHash"
    Write-FixtureFile '.github/workflows/android.yml' 'check_open_source_readiness.ps1'
    Write-FixtureFile '.github/workflows/release.yml' 'verify_release_apk.ps1 generate_release_manifest.ps1'

    $manifest = [ordered]@{
        schema = 'blindassist_public_release_assets_v1'
        assets = @(
            [ordered]@{
                path = $modelPath
                size_bytes = (Get-Item -LiteralPath $modelFile).Length
                sha256 = $modelHash
                license_expression = 'Apache-2.0'
                upstream_url = 'https://example.invalid/model'
                notice_path = 'THIRD_PARTY_NOTICES.md'
                packaged_in_default_app = $true
            },
            [ordered]@{
                path = $labelsPath
                size_bytes = (Get-Item -LiteralPath $labelsFile).Length
                sha256 = $labelsHash
                license_expression = 'NOASSERTION'
                upstream_url = 'https://example.invalid/labels'
                notice_path = 'THIRD_PARTY_NOTICES.md'
                packaged_in_default_app = $true
            }
        )
    }
    Write-FixtureFile 'configs/public_release_assets.json' ($manifest | ConvertTo-Json -Depth 5)

    Assert-Check $true 'complete-public-project'
    Write-FixtureFile $modelPath 'tampered-model-bytes'
    Assert-Check $false 'asset-hash-drift'
}
finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
