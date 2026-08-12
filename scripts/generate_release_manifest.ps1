param(
    [Parameter(Mandatory = $true)]
    [string[]]$ArtifactPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$VerificationJsonPath,
    [string]$SourceCommit
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $repoRoot $Path
}

if (-not $SourceCommit) {
    $SourceCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $SourceCommit) { throw 'Could not resolve source commit.' }
}

$buildGradle = Get-Content -Raw -LiteralPath (Join-Path $repoRoot 'app/build.gradle.kts') -Encoding UTF8
$versionCodeMatch = [regex]::Match($buildGradle, 'versionCode\s*=\s*(\d+)')
$versionNameMatch = [regex]::Match($buildGradle, 'versionName\s*=\s*"([^"]+)"')
if (-not $versionCodeMatch.Success -or -not $versionNameMatch.Success) {
    throw 'Could not read versionCode and versionName from app/build.gradle.kts.'
}
$versionCode = [int]$versionCodeMatch.Groups[1].Value
$versionName = $versionNameMatch.Groups[1].Value

$verification = $null
if ($VerificationJsonPath) {
    $resolvedVerification = Resolve-RepoPath $VerificationJsonPath
    if (-not (Test-Path -LiteralPath $resolvedVerification -PathType Leaf)) {
        throw "Verification JSON not found: $resolvedVerification"
    }
    $verification = Get-Content -Raw -LiteralPath $resolvedVerification -Encoding UTF8 | ConvertFrom-Json
}

$artifacts = @()
$seenNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($requestedPath in $ArtifactPath) {
    $resolvedArtifact = Resolve-RepoPath $requestedPath
    if (-not (Test-Path -LiteralPath $resolvedArtifact -PathType Leaf)) {
        throw "Release artifact not found: $resolvedArtifact"
    }
    $resolvedArtifact = (Resolve-Path -LiteralPath $resolvedArtifact).Path
    $name = [System.IO.Path]::GetFileName($resolvedArtifact)
    if (-not $seenNames.Add($name)) { throw "Duplicate release artifact name: $name" }
    $hash = (Get-FileHash -LiteralPath $resolvedArtifact -Algorithm SHA256).Hash
    if ($verification -and @($ArtifactPath).Count -eq 1 -and
        -not $hash.Equals([string]$verification.sha256, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Verification SHA-256 does not match release artifact $name."
    }
    $artifacts += [pscustomobject][ordered]@{
        fileName = $name
        sizeBytes = (Get-Item -LiteralPath $resolvedArtifact).Length
        sha256 = $hash
    }
}

$resolvedOutput = Resolve-RepoPath $OutputDirectory
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
$generatedAt = (Get-Date).ToUniversalTime().ToString('o')
$manifest = [pscustomobject][ordered]@{
    schema = 'blindassist_release_manifest_v1'
    generatedAt = $generatedAt
    sourceCommit = $SourceCommit
    versionCode = $versionCode
    versionName = $versionName
    artifactKind = 'debug-signed-evaluation'
    artifacts = $artifacts
    evidenceBoundary = 'Build and static artifact verification are not real-device accuracy, user-outcome, production-signing, or safety evidence.'
}

$manifestPath = Join-Path $resolvedOutput 'release-manifest.json'
$checksumsPath = Join-Path $resolvedOutput 'SHA256SUMS'
$summaryPath = Join-Path $resolvedOutput 'RELEASE_VERIFICATION.md'
[System.IO.File]::WriteAllText(
    $manifestPath,
    ($manifest | ConvertTo-Json -Depth 5) + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)
$checksumLines = @($artifacts | ForEach-Object { "$($_.sha256)  $($_.fileName)" })
[System.IO.File]::WriteAllText(
    $checksumsPath,
    ($checksumLines -join [Environment]::NewLine) + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)
$artifactRows = @($artifacts | ForEach-Object { "| ``$($_.fileName)`` | $($_.sizeBytes) | ``$($_.sha256)`` |" })
$summary = @(
    "# BlindAssist v$versionName verification",
    '',
    "- Source commit: ``$SourceCommit``",
    "- Version: ``versionCode=$versionCode``, ``versionName=$versionName``",
    '- Artifact type: debug-signed evaluation build',
    '',
    '| Artifact | Bytes | SHA-256 |',
    '| --- | ---: | --- |',
    $artifactRows,
    '',
    '## Evidence boundary',
    '',
    'The artifact is not production-signed, safety-certified, or evidence of real-world assistive accuracy. Static package checks do not replace device validation; untested behavior remains `UNKNOWN`.'
) -join [Environment]::NewLine
[System.IO.File]::WriteAllText($summaryPath, $summary + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))

[pscustomobject][ordered]@{
    status = 'PASS'
    manifestPath = $manifestPath
    checksumsPath = $checksumsPath
    summaryPath = $summaryPath
    manifest = $manifest
} | ConvertTo-Json -Depth 6
