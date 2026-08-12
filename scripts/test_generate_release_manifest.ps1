$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$generator = Join-Path $PSScriptRoot 'generate_release_manifest.ps1'
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("blindassist-release-manifest-" + [guid]::NewGuid().ToString('N'))

try {
    New-Item -ItemType Directory -Force -Path $testRoot | Out-Null
    $artifact = Join-Path $testRoot 'BlindAssist-test.apk'
    [System.IO.File]::WriteAllText($artifact, 'artifact-bytes', [System.Text.UTF8Encoding]::new($false))
    $hash = (Get-FileHash -LiteralPath $artifact -Algorithm SHA256).Hash
    $verificationPath = Join-Path $testRoot 'verification.json'
    [System.IO.File]::WriteAllText(
        $verificationPath,
        ([ordered]@{ sha256 = $hash } | ConvertTo-Json) + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    $output = Join-Path $testRoot 'output'
    & $generator -ArtifactPath $artifact -OutputDirectory $output -VerificationJsonPath $verificationPath -SourceCommit ('A' * 40) | Out-Null

    $manifestPath = Join-Path $output 'release-manifest.json'
    $checksumsPath = Join-Path $output 'SHA256SUMS'
    $summaryPath = Join-Path $output 'RELEASE_VERIFICATION.md'
    foreach ($path in @($manifestPath, $checksumsPath, $summaryPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing generated release file: $path" }
    }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schema -ne 'blindassist_release_manifest_v1' -or
        $manifest.sourceCommit -ne ('A' * 40) -or
        $manifest.artifacts[0].sha256 -ne $hash) {
        throw 'Generated release manifest identity is incorrect.'
    }
    $checksums = Get-Content -Raw -LiteralPath $checksumsPath -Encoding UTF8
    if ($checksums -notmatch [regex]::Escape("$hash  BlindAssist-test.apk")) {
        throw 'Generated SHA256SUMS is incorrect.'
    }
    Write-Host 'PASS: release-manifest-and-checksums'

    [System.IO.File]::WriteAllText(
        $verificationPath,
        ([ordered]@{ sha256 = ('0' * 64) } | ConvertTo-Json) + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    $mismatchFailed = $false
    try {
        & $generator -ArtifactPath $artifact -OutputDirectory $output -VerificationJsonPath $verificationPath -SourceCommit ('A' * 40) | Out-Null
    }
    catch {
        $mismatchFailed = $true
    }
    if (-not $mismatchFailed) { throw 'Mismatched verification hash was expected to fail.' }
    Write-Host 'PASS: verification-hash-mismatch'
}
finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
