param(
    [string]$CandidateApk = "apps\candidates\npu-candidate\build\outputs\apk\debug\npu-candidate-debug.apk",
    [string]$BaselinePackage = "com.linnan.blindassist",
    [string]$CandidatePackage = "com.linnan.blindassist.npu.candidate",
    [string]$Device,
    [string]$AndroidSdkRoot,
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $PSScriptRoot "..\$Path"
}

function Invoke-Adb([string[]]$Arguments, [switch]$AllowFailure) {
    $output = (& $script:Adb -s $script:DeviceId @Arguments 2>&1) -join "`n"
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "adb $($Arguments -join ' ') failed ($exitCode):`n$output"
    }
    return [pscustomobject]@{ exitCode = $exitCode; output = $output.Trim() }
}

function Get-PackageSnapshot([string]$PackageName) {
    $path = Invoke-Adb @("shell", "pm", "path", $PackageName) -AllowFailure
    if ($path.exitCode -ne 0 -or $path.output -notmatch "^package:") { return $null }
    $dump = (Invoke-Adb @("shell", "dumpsys", "package", $PackageName)).output
    function Match-One([string]$Pattern) {
        $match = [regex]::Match($dump, $Pattern, [Text.RegularExpressions.RegexOptions]::Multiline)
        if ($match.Success) { return $match.Groups[1].Value.Trim() }
        return $null
    }
    return [ordered]@{
        packageName = $PackageName
        apkPath = $path.output.Substring("package:".Length)
        versionCode = Match-One "^\s*versionCode=(\d+)"
        versionName = Match-One "^\s*versionName=([^\r\n]+)"
        appId = Match-One "^\s*appId=(\d+)"
        firstInstallTime = Match-One "^\s*firstInstallTime=([^\r\n]+)"
        lastUpdateTime = Match-One "^\s*lastUpdateTime=([^\r\n]+)"
    }
}

function Get-AppDataFingerprint([string]$PackageName) {
    $listing = Invoke-Adb @(
        "shell", "run-as", $PackageName,
        "find", "shared_prefs", "files", "-type", "f", "-print"
    ) -AllowFailure
    if ($listing.exitCode -ne 0) {
        return [ordered]@{ available = $false; sha256 = $null; reason = $listing.output }
    }
    $paths = @(($listing.output -split "`r?`n") | Where-Object { $_ } | Sort-Object)
    $rows = foreach ($path in $paths) {
        $digest = Invoke-Adb @(
            "shell", "run-as", $PackageName, "sha256sum", $path
        ) -AllowFailure
        if ($digest.exitCode -ne 0 -or $digest.output -notmatch "^([0-9a-fA-F]{64})\s+") {
            return [ordered]@{
                available = $false
                sha256 = $null
                reason = "Failed to hash ${path}: $($digest.output)"
            }
        }
        "$($Matches[1].ToUpperInvariant()) $path"
    }
    $canonical = $rows -join "`n"
    $bytes = [Text.Encoding]::UTF8.GetBytes($canonical)
    $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))
    return [ordered]@{
        available = $true
        sha256 = $hash
        fileCount = $paths.Count
        files = $rows
    }
}

function Start-Package([string]$PackageName) {
    Invoke-Adb @("shell", "am", "force-stop", $PackageName) | Out-Null
    $start = Invoke-Adb @(
        "shell", "am", "start", "-W", "-n",
        "$PackageName/com.linnan.blindassist.MainActivity"
    )
    Start-Sleep -Seconds 2
    $pidResult = Invoke-Adb @("shell", "pidof", $PackageName) -AllowFailure
    if ($pidResult.exitCode -ne 0 -or -not $pidResult.output) {
        throw "$PackageName did not remain running after launch:`n$($start.output)"
    }
    $processId = ($pidResult.output -split "\s+")[0]
    $logs = (Invoke-Adb @("logcat", "--pid=$processId", "-d", "-t", "1200")).output
    return [ordered]@{
        pid = $processId
        startOutput = $start.output
        logs = $logs
    }
}

function Assert-SnapshotUnchanged($Before, $After) {
    foreach ($field in @("packageName", "apkPath", "versionCode", "versionName", "appId", "firstInstallTime", "lastUpdateTime")) {
        if ($Before[$field] -ne $After[$field]) {
            throw "Baseline package changed at ${field}: '$($Before[$field])' -> '$($After[$field])'"
        }
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$resolvedApk = (Resolve-Path -LiteralPath (Resolve-RepoPath $CandidateApk)).Path
if (-not $AndroidSdkRoot) { $AndroidSdkRoot = Join-Path $repoRoot ".android-sdk" }
$script:Adb = Join-Path (Resolve-RepoPath $AndroidSdkRoot) "platform-tools\adb.exe"
if (-not (Test-Path -LiteralPath $script:Adb)) { throw "adb not found: $script:Adb" }

$devices = (& $script:Adb devices) |
    Select-Object -Skip 1 |
    Where-Object { $_ -match "\sdevice$" } |
    ForEach-Object { ($_ -split "\s+")[0] }
if ($Device) {
    if ($Device -notin $devices) { throw "Requested device is not ready: $Device" }
    $script:DeviceId = $Device
} elseif ($devices.Count -eq 1) {
    $script:DeviceId = $devices[0]
} else {
    throw "Expected exactly one ready device or explicit -Device; found $($devices.Count)"
}

if (-not $OutputDir) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $repoRoot "artifacts.local\evidence\npu-candidate-acceptance\$stamp"
}
$resolvedOutput = [IO.Path]::GetFullPath((Resolve-RepoPath $OutputDir))
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null

$model = (Invoke-Adb @("shell", "getprop", "ro.product.model")).output
$soc = (Invoke-Adb @("shell", "getprop", "ro.soc.model")).output
if ($model -ne "SM-S9280" -or $soc -ne "SM8650") {
    throw "Candidate acceptance is frozen to SM-S9280/SM8650, got $model/$soc"
}

$baselineBefore = Get-PackageSnapshot $BaselinePackage
if ($null -eq $baselineBefore) { throw "Baseline package is not installed: $BaselinePackage" }
if ($null -ne (Get-PackageSnapshot $CandidatePackage)) {
    throw "Candidate package is already installed; remove only that package before this one-shot acceptance."
}
$dataBefore = Get-AppDataFingerprint $BaselinePackage
$candidateInstalled = $false
$candidateRun = $null
$rollbackRun = $null
$failure = $null

try {
    $install = Invoke-Adb @("install", $resolvedApk)
    $candidateInstalled = $true
    if ($install.output -notmatch "Success") { throw "Candidate install did not report Success." }

    $candidateSnapshot = Get-PackageSnapshot $CandidatePackage
    $baselineDuring = Get-PackageSnapshot $BaselinePackage
    if ($null -eq $candidateSnapshot) { throw "Candidate package is missing after install." }
    Assert-SnapshotUnchanged $baselineBefore $baselineDuring
    if ($candidateSnapshot.appId -eq $baselineBefore.appId) {
        throw "Candidate and baseline unexpectedly share the same Android UID."
    }

    $candidateRun = Start-Package $CandidatePackage
    if ($candidateRun.logs -notmatch "Detector ready backend=qualcomm_qnn_htp") {
        throw "Candidate did not prove QNN HTP detector readiness."
    }
    if ($candidateRun.logs -match "(?i)falling back to CPU|backend=cpu_xnnpack") {
        throw "Candidate logs show a forbidden CPU fallback."
    }

    Invoke-Adb @("shell", "am", "force-stop", $CandidatePackage) | Out-Null
    $uninstall = Invoke-Adb @("uninstall", $CandidatePackage)
    $candidateInstalled = $false
    if ($uninstall.output -notmatch "Success") { throw "Candidate uninstall did not report Success." }

    $baselineAfter = Get-PackageSnapshot $BaselinePackage
    Assert-SnapshotUnchanged $baselineBefore $baselineAfter
    $dataAfter = Get-AppDataFingerprint $BaselinePackage
    if ($dataBefore.available -and
        (-not $dataAfter.available -or $dataBefore.sha256 -ne $dataAfter.sha256)) {
        throw "Baseline app-data fingerprint changed during candidate install/uninstall."
    }
    if ($null -ne (Get-PackageSnapshot $CandidatePackage)) {
        throw "Candidate package remains installed after rollback."
    }

    $rollbackRun = Start-Package $BaselinePackage
    if ($rollbackRun.logs -match "Detector ready backend=qualcomm_qnn_htp") {
        throw "Baseline unexpectedly reported the candidate QNN backend after rollback."
    }

    $summary = [ordered]@{
        schema = "blindassist_npu_candidate_acceptance_v1"
        disposition = "PASS_CANDIDATE_INSTALL_COEXISTENCE_AND_ROLLBACK"
        device = [ordered]@{ serial = $script:DeviceId; model = $model; soc = $soc }
        candidateApk = [ordered]@{
            path = $resolvedApk
            sizeBytes = (Get-Item -LiteralPath $resolvedApk).Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedApk).Hash
        }
        baselineBefore = $baselineBefore
        baselineDataBefore = $dataBefore
        candidate = $candidateSnapshot
        candidateStart = $candidateRun.startOutput
        baselineAfter = $baselineAfter
        baselineDataAfter = $dataAfter
        rollbackStart = $rollbackRun.startOutput
        rollbackBackendEvidence = if ($rollbackRun.logs -match "Detector ready backend=cpu_xnnpack") {
            "RUNTIME_CPU_MARKER"
        } else {
            "PREEXISTING_APK_PACKAGE_DATA_AND_LAUNCH_INVARIANTS"
        }
    }
} catch {
    $failure = $_.Exception.Message
    $summary = [ordered]@{
        schema = "blindassist_npu_candidate_acceptance_v1"
        disposition = "HOLD_CANDIDATE_ACCEPTANCE_FAILED"
        device = [ordered]@{ serial = $script:DeviceId; model = $model; soc = $soc }
        failure = $failure
    }
} finally {
    if ($candidateInstalled) {
        Invoke-Adb @("uninstall", $CandidatePackage) -AllowFailure | Out-Null
    }
    if ($candidateRun) {
        Set-Content -LiteralPath (Join-Path $resolvedOutput "candidate-logcat.txt") `
            -Value $candidateRun.logs -Encoding UTF8
    }
    if ($rollbackRun) {
        Set-Content -LiteralPath (Join-Path $resolvedOutput "rollback-baseline-logcat.txt") `
            -Value $rollbackRun.logs -Encoding UTF8
    }
    $summary | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $resolvedOutput "summary.json") -Encoding UTF8
}

$summary | ConvertTo-Json -Depth 8
if ($failure) { exit 1 }
