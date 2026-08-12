param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$resolvedRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$rootPrefix = $resolvedRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
$failures = [Collections.Generic.List[string]]::new()
$manifest = $null

function Resolve-PublicPath([string]$RelativePath) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot $RelativePath))
    if (-not $candidate.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Public asset path escapes the repository: $RelativePath"
    }
    return $candidate
}

function Require-File([string]$RelativePath) {
    $path = Resolve-PublicPath $RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $failures.Add("Missing open-source maintenance file: $RelativePath")
    }
    return $path
}

$requiredFiles = @(
    'README.md',
    'LICENSE',
    'THIRD_PARTY_NOTICES.md',
    'CONTRIBUTING.md',
    'SECURITY.md',
    'CODE_OF_CONDUCT.md',
    'GOVERNANCE.md',
    'CITATION.cff',
    'docs/OPEN_SOURCE_PUBLIC_VALUE.md',
    'docs/MODEL_CARD.md',
    '.github/CODEOWNERS',
    '.github/dependabot.yml',
    '.github/ISSUE_TEMPLATE/bug_report.yml',
    '.github/ISSUE_TEMPLATE/feature_request.yml',
    '.github/pull_request_template.md',
    '.github/workflows/android.yml',
    '.github/workflows/release.yml',
    'configs/public_release_assets.json'
)
foreach ($relativePath in $requiredFiles) {
    Require-File $relativePath | Out-Null
}

$manifestPath = Resolve-PublicPath 'configs/public_release_assets.json'
$verifiedAssets = @()
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    try {
        $manifest = Get-Content -Raw -LiteralPath $manifestPath -Encoding UTF8 | ConvertFrom-Json
        if ($manifest.schema -ne 'blindassist_public_release_assets_v1') {
            $failures.Add("Unexpected public asset manifest schema: $($manifest.schema)")
        }
        if (-not $manifest.assets -or @($manifest.assets).Count -eq 0) {
            $failures.Add('Public asset manifest must declare at least one asset.')
        }
        foreach ($asset in @($manifest.assets)) {
            $relativeAssetPath = [string]$asset.path
            $assetPath = Resolve-PublicPath $relativeAssetPath
            if (-not (Test-Path -LiteralPath $assetPath -PathType Leaf)) {
                $failures.Add("Declared public asset is missing: $relativeAssetPath")
                continue
            }
            $actualSize = (Get-Item -LiteralPath $assetPath).Length
            $actualHash = (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash
            if ($actualSize -ne [long]$asset.size_bytes) {
                $failures.Add("Public asset size mismatch for $relativeAssetPath. Expected $($asset.size_bytes), got $actualSize.")
            }
            if (-not $actualHash.Equals([string]$asset.sha256, [StringComparison]::OrdinalIgnoreCase)) {
                $failures.Add("Public asset SHA-256 mismatch for $relativeAssetPath. Expected $($asset.sha256), got $actualHash.")
            }
            if (-not [string]$asset.license_expression -or -not [string]$asset.upstream_url -or -not [string]$asset.notice_path) {
                $failures.Add("Public asset provenance is incomplete for $relativeAssetPath.")
            }
            else {
                $noticePath = Resolve-PublicPath ([string]$asset.notice_path)
                if (-not (Test-Path -LiteralPath $noticePath -PathType Leaf)) {
                    $failures.Add("Public asset notice is missing for ${relativeAssetPath}: $($asset.notice_path)")
                }
                elseif ((Get-Content -Raw -LiteralPath $noticePath -Encoding UTF8) -notmatch [regex]::Escape($relativeAssetPath)) {
                    $failures.Add("Public asset notice does not name $relativeAssetPath.")
                }
            }
            $verifiedAssets += [pscustomobject][ordered]@{
                path = $relativeAssetPath
                sizeBytes = $actualSize
                sha256 = $actualHash
                licenseExpression = [string]$asset.license_expression
            }
        }
    }
    catch {
        $failures.Add("Could not validate public asset manifest: $($_.Exception.Message)")
    }
}

$readmePath = Resolve-PublicPath 'README.md'
if (Test-Path -LiteralPath $readmePath -PathType Leaf) {
    $readme = Get-Content -Raw -LiteralPath $readmePath -Encoding UTF8
    foreach ($requiredLink in @('CONTRIBUTING.md', 'GOVERNANCE.md', 'docs/MODEL_CARD.md', 'SECURITY.md')) {
        if ($readme -notmatch [regex]::Escape($requiredLink)) {
            $failures.Add("README.md must link to $requiredLink.")
        }
    }
}

$modelCardPath = Resolve-PublicPath 'docs/MODEL_CARD.md'
if ((Test-Path -LiteralPath $modelCardPath -PathType Leaf) -and $null -ne $manifest) {
    $modelCard = Get-Content -Raw -LiteralPath $modelCardPath -Encoding UTF8
    foreach ($asset in @($manifest.assets)) {
        if ([bool]$asset.packaged_in_default_app) {
            if ($modelCard -notmatch [regex]::Escape([string]$asset.path) -or
                $modelCard -notmatch [regex]::Escape([string]$asset.sha256)) {
                $failures.Add("Default App asset is not identity-bound in docs/MODEL_CARD.md: $($asset.path)")
            }
        }
    }
}

$androidWorkflowPath = Resolve-PublicPath '.github/workflows/android.yml'
if (Test-Path -LiteralPath $androidWorkflowPath -PathType Leaf) {
    $androidWorkflow = Get-Content -Raw -LiteralPath $androidWorkflowPath -Encoding UTF8
    if ($androidWorkflow -notmatch 'check_open_source_readiness\.ps1') {
        $failures.Add('Android CI must run scripts/check_open_source_readiness.ps1.')
    }
}

$releaseWorkflowPath = Resolve-PublicPath '.github/workflows/release.yml'
if (Test-Path -LiteralPath $releaseWorkflowPath -PathType Leaf) {
    $releaseWorkflow = Get-Content -Raw -LiteralPath $releaseWorkflowPath -Encoding UTF8
    foreach ($requiredReleaseStep in @('verify_release_apk.ps1', 'generate_release_manifest.ps1')) {
        if ($releaseWorkflow -notmatch [regex]::Escape($requiredReleaseStep)) {
            $failures.Add("Release workflow must run $requiredReleaseStep.")
        }
    }
}

if ($failures.Count -gt 0) {
    throw "Open-source readiness failed:`n- $($failures -join "`n- ")"
}

[pscustomobject][ordered]@{
    status = 'PASS'
    schema = 'blindassist_open_source_readiness_v1'
    requiredFiles = $requiredFiles.Count
    verifiedAssets = $verifiedAssets
} | ConvertTo-Json -Depth 5
