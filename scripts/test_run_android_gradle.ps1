[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workspaceRoot = Split-Path -Parent $repoRoot
$launcher = Join-Path $PSScriptRoot "run_android_gradle.ps1"

Push-Location $workspaceRoot
try {
    $output = @(& $launcher -PreflightOnly)
    if ($LASTEXITCODE -ne 0) {
        throw "Environment preflight returned $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
$jsonLine = $output |
    Where-Object { $_ -is [string] -and $_.StartsWith('{') } |
    Select-Object -Last 1
if ([string]::IsNullOrWhiteSpace($jsonLine)) {
    throw "Environment preflight did not emit its JSON summary."
}
$summary = $jsonLine | ConvertFrom-Json
if (
    $summary.status -ne "ENVIRONMENT_READY" -or
    $summary.repo_root -ne $repoRoot -or
    -not ([string]$summary.java_version).StartsWith("17.") -or
    $summary.compile_sdk -ne "35" -or
    $summary.gradle_version -ne "8.10.2"
) {
    throw "Environment preflight summary did not match the project contract."
}

$helpOutput = @(& $launcher help --offline)
if (
    $LASTEXITCODE -ne 0 -or
    -not ($helpOutput -contains "GRADLE_COMMAND_COMPLETE: exit_code=0")
) {
    throw "Gradle arguments were not forwarded through the launcher."
}

$missingJava = Join-Path (
    Join-Path $repoRoot "artifacts.local\tmp"
) ("missing-jdk-{0}" -f [guid]::NewGuid().ToString("N"))
$pwsh = Join-Path $PSHOME "pwsh.exe"
if (-not (Test-Path -LiteralPath $pwsh -PathType Leaf)) {
    $pwsh = (Get-Process -Id $PID).Path
}
$escapedLauncher = $launcher.Replace("'", "''")
$escapedMissingJava = $missingJava.Replace("'", "''")
$command = (
    '$env:BLINDASSIST_JAVA_HOME=''' + $escapedMissingJava +
    '''; & ''' + $escapedLauncher +
    ''' -PreflightOnly; exit $LASTEXITCODE'
)
$failureOutput = @(& $pwsh -NoProfile -Command $command 2>&1)
$failureExitCode = $LASTEXITCODE
if (
    $failureExitCode -ne 20 -or
    -not (($failureOutput -join [Environment]::NewLine) -match "ENV_BLOCKED:")
) {
    throw (
        "Invalid explicit JDK was not rejected deterministically. " +
        "exit=$failureExitCode output='$($failureOutput -join ' | ')'"
    )
}

Write-Output "PASS: Android Gradle environment launcher"
