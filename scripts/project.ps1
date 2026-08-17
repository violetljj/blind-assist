[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet("doctor", "bootstrap", "test", "run", "rebuild")]
    [string]$Command = "doctor",
    [string]$PythonScript,
    [string[]]$TargetArguments,
    [string[]]$GradleArguments,
    [switch]$RequireDevice,
    [string]$AndroidSerial
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "E:\codex-tools\bin\blindassist-python.cmd"
$GradleEntry = Join-Path $PSScriptRoot "run_android_gradle.ps1"
$StructureGate = Join-Path $PSScriptRoot "check_project_structure.ps1"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$NativeArguments
    )
    & $FilePath @NativeArguments
    if ($LASTEXITCODE -ne 0) { throw "ENV_BLOCKED: '$FilePath' exited with $LASTEXITCODE." }
}

function Invoke-Doctor {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "ENV_BLOCKED: missing evidence-bound Python launcher: $Python"
    }
    if (-not (Test-Path -LiteralPath $GradleEntry -PathType Leaf)) {
        throw "ENV_BLOCKED: missing Android entry: $GradleEntry"
    }
    Invoke-NativeChecked $Python -c "import platform,numpy,cv2; assert platform.python_version() == '3.11.9'; print('PASS BlindAssist Python', platform.python_version(), 'numpy', numpy.__version__, 'opencv', cv2.__version__)"
    & $GradleEntry -PreflightOnly -RequireDevice:$RequireDevice -AndroidSerial $AndroidSerial
    if ($LASTEXITCODE -ne 0) { throw "ENV_BLOCKED: Android/Gradle preflight exited with $LASTEXITCODE." }
    Write-Host "PASS protected toolchains are ready; no project environment was modified."
}

function Resolve-RepoScript([string]$Path) {
    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    $prefix = $RepoRoot.TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "REFUSED: Python script is outside the repository: $resolved"
    }
    return $resolved
}

switch ($Command) {
    "doctor" { Invoke-Doctor }
    "bootstrap" {
        Write-Host "PASS BlindAssist uses protected, evidence-bound toolchains; bootstrap is a non-mutating readiness check."
        Invoke-Doctor
    }
    "test" {
        Invoke-Doctor
        & $StructureGate
        if ($LASTEXITCODE -ne 0) { throw "TEST_FAILED: project structure gate exited with $LASTEXITCODE." }
    }
    "run" {
        $hasPython = -not [string]::IsNullOrWhiteSpace($PythonScript)
        $hasGradle = $null -ne $GradleArguments -and $GradleArguments.Count -gt 0
        if ($hasPython -eq $hasGradle) { throw "USAGE: run requires exactly one of -PythonScript or -GradleArguments." }
        if ($hasPython) {
            Invoke-Doctor
            $script = Resolve-RepoScript $PythonScript
            Invoke-NativeChecked $Python $script @TargetArguments
        }
        else {
            & $GradleEntry -RequireDevice:$RequireDevice -AndroidSerial $AndroidSerial @GradleArguments
            if ($LASTEXITCODE -ne 0) { throw "RUN_FAILED: Android Gradle entry exited with $LASTEXITCODE." }
        }
    }
    "rebuild" {
        throw "REFUSED: BlindAssist toolchains and evidence environments are protected. Diagnose with 'project.ps1 doctor'; rebuild only through the owning, reviewed environment procedure."
    }
}
