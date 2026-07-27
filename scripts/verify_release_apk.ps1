param(
    [Parameter(Mandatory = $true)]
    [string]$ApkPath,
    [int]$ExpectedVersionCode,
    [string]$ExpectedVersionName,
    [string]$ExpectedPackageName = "com.linnan.blindassist",
    [switch]$RequireReleaseEvidence,
    [switch]$AllowLegacyExtractedNativeLibraries,
    [string]$AndroidSdkRoot
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $PSScriptRoot "..\$Path")
}

function Get-LatestBuildTools([string]$SdkRoot) {
    $buildTools = Join-Path $SdkRoot "build-tools"
    if (-not (Test-Path -LiteralPath $buildTools)) {
        throw "Android build-tools not found under $buildTools"
    }
    return Get-ChildItem -LiteralPath $buildTools -Directory |
        Sort-Object { try { [version]$_.Name } catch { [version]"0.0" } } -Descending |
        Select-Object -First 1
}

function Assert-Contains([string]$Text, [string]$Pattern, [string]$Message) {
    if ($Text -notmatch $Pattern) {
        throw $Message
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not $AndroidSdkRoot) {
    $AndroidSdkRoot = Join-Path $repoRoot ".android-sdk"
}
$resolvedApk = Resolve-RepoPath $ApkPath
if (-not (Test-Path -LiteralPath $resolvedApk)) {
    throw "APK not found: $resolvedApk"
}
$resolvedApk = (Resolve-Path -LiteralPath $resolvedApk).Path

$tools = Get-LatestBuildTools (Resolve-RepoPath $AndroidSdkRoot)
$aapt = Join-Path $tools.FullName "aapt.exe"
$apksigner = Join-Path $tools.FullName "apksigner.bat"
if (-not (Test-Path -LiteralPath $aapt)) {
    throw "aapt.exe not found in $($tools.FullName)"
}
if (-not (Test-Path -LiteralPath $apksigner)) {
    throw "apksigner.bat not found in $($tools.FullName)"
}

$badging = (& $aapt dump badging $resolvedApk) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "aapt dump badging failed"
}

$packageMatch = [regex]::Match($badging, "package: name='([^']+)' versionCode='([^']+)' versionName='([^']+)'")
if (-not $packageMatch.Success) {
    throw "Could not parse APK package metadata."
}

$actualPackage = $packageMatch.Groups[1].Value
$actualVersionCode = [int]$packageMatch.Groups[2].Value
$actualVersionName = $packageMatch.Groups[3].Value

if ($actualPackage -ne $ExpectedPackageName) {
    throw "Package mismatch. Expected $ExpectedPackageName, got $actualPackage"
}
if ($PSBoundParameters.ContainsKey("ExpectedVersionCode") -and $actualVersionCode -ne $ExpectedVersionCode) {
    throw "versionCode mismatch. Expected $ExpectedVersionCode, got $actualVersionCode"
}
if ($ExpectedVersionName -and $actualVersionName -ne $ExpectedVersionName) {
    throw "versionName mismatch. Expected $ExpectedVersionName, got $actualVersionName"
}

$signature = (& $apksigner verify --print-certs $resolvedApk) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "apksigner verify failed"
}
Assert-Contains $signature "Signer #1 certificate SHA-256 digest:" "Missing signing certificate SHA-256 digest."

$alignmentOutput = (& (Join-Path $PSScriptRoot "verify_apk_16kb.ps1") `
    -ArtifactPath $resolvedApk `
    -AllowLegacyExtractedNativeLibraries:$AllowLegacyExtractedNativeLibraries `
    -AndroidSdkRoot $AndroidSdkRoot) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "16KB APK verification failed."
}
$alignmentEvidence = $alignmentOutput | ConvertFrom-Json

$releaseEvidence = [ordered]@{
    manifestMergerReleaseReport = Test-Path -LiteralPath (Join-Path $repoRoot "app\build\outputs\logs\manifest-merger-release-report.txt")
    releaseApkDirectory = Test-Path -LiteralPath (Join-Path $repoRoot "app\build\outputs\apk\release")
    releaseMappingDirectory = Test-Path -LiteralPath (Join-Path $repoRoot "app\build\outputs\mapping\release")
    releaseGradleConfigMentionsMinify = $false
    releaseMinifyEnabled = $null
    releaseShrinkResources = $null
    releaseSigningConfigMentioned = $false
}

$buildGradle = Get-Content -LiteralPath (Join-Path $repoRoot "app\build.gradle.kts") -Raw
$releaseBlockMatch = [regex]::Match($buildGradle, "release\s*\{(?<body>[\s\S]*?)\n\s*\}")
if ($releaseBlockMatch.Success) {
    $releaseBlock = $releaseBlockMatch.Groups["body"].Value
    $minifyMatch = [regex]::Match($releaseBlock, "isMinifyEnabled\s*=\s*(true|false)")
    $shrinkMatch = [regex]::Match($releaseBlock, "isShrinkResources\s*=\s*(true|false)")
    $releaseEvidence.releaseGradleConfigMentionsMinify = $minifyMatch.Success
    if ($minifyMatch.Success) {
        $releaseEvidence.releaseMinifyEnabled = [bool]::Parse($minifyMatch.Groups[1].Value)
    }
    if ($shrinkMatch.Success) {
        $releaseEvidence.releaseShrinkResources = [bool]::Parse($shrinkMatch.Groups[1].Value)
    }
    $releaseEvidence.releaseSigningConfigMentioned = $releaseBlock -match "signingConfig"
}

if ($RequireReleaseEvidence) {
    if (-not $releaseEvidence.manifestMergerReleaseReport) {
        throw "Missing release manifest merger report."
    }
    if (-not $releaseEvidence.releaseGradleConfigMentionsMinify) {
        throw "Missing release shrink/minify configuration evidence in app/build.gradle.kts."
    }
}

$result = [ordered]@{
    apk = $resolvedApk
    sizeBytes = (Get-Item -LiteralPath $resolvedApk).Length
    sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedApk).Hash
    packageName = $actualPackage
    versionCode = $actualVersionCode
    versionName = $actualVersionName
    signingCertificate = ($signature -split "`n" | Where-Object { $_ -match "Signer #1 certificate .*digest:" })
    alignment16Kb = $alignmentEvidence
    releaseEvidence = $releaseEvidence
}

$result | ConvertTo-Json -Depth 5
