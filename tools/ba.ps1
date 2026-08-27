[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'doctor', 'smoke', 'run', 'clean')]
    [string]$Command = 'doctor',
    [Parameter(Position = 1)]
    [ValidateSet('base', 'research-dtr-r0', 'android', 'device', 'export')]
    [string]$Profile = 'base',
    [string]$Python,
    [string]$Docker,
    [string]$EventInput,
    [string]$ResultOutput,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$ActiveRoot = Join-Path $RepoRoot 'research/active/dtr-r0'
$LocalConfigPath = Join-Path $RepoRoot 'config/local.toml'

function Stop-Ba {
    param([string]$Code, [string]$Detail, [string]$Fix)
    Write-Output $Code
    Write-Output "profile: $Profile"
    Write-Output "detail: $Detail"
    Write-Output "fix: $Fix"
    exit 2
}

function Invoke-NativeChecked {
    param([string]$FilePath, [string[]]$NativeArguments)
    & $FilePath @NativeArguments
    if ($LASTEXITCODE -ne 0) {
        Stop-Ba 'BA_COMMAND_FAILED' "'$FilePath' exited with $LASTEXITCODE" "rerun the reported command after fixing its first error"
    }
}

function Read-LocalConfig {
    $values = @{}
    if (-not (Test-Path -LiteralPath $LocalConfigPath -PathType Leaf)) { return $values }
    foreach ($line in Get-Content -LiteralPath $LocalConfigPath) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed.StartsWith('[')) { continue }
        if ($trimmed -match '^([A-Za-z0-9_-]+)\s*=\s*"(.*)"\s*$') {
            $values[$Matches[1]] = $Matches[2]
        }
    }
    return $values
}

function Resolve-ConfiguredPath {
    param([string]$Explicit, [string]$ConfigKey, [string]$EnvironmentKey, [string]$Default)
    $local = Read-LocalConfig
    $value = $Explicit
    if ([string]::IsNullOrWhiteSpace($value) -and $local.ContainsKey($ConfigKey)) { $value = $local[$ConfigKey] }
    if ([string]::IsNullOrWhiteSpace($value)) { $value = [Environment]::GetEnvironmentVariable($EnvironmentKey) }
    if ([string]::IsNullOrWhiteSpace($value)) { $value = $Default }
    if ([string]::IsNullOrWhiteSpace($value)) { return $null }
    if ([System.IO.Path]::IsPathRooted($value)) { return [System.IO.Path]::GetFullPath($value) }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $value))
}

function Resolve-ResearchPython {
    $candidate = Resolve-ConfiguredPath $Python 'research_python' 'BLINDASSIST_RESEARCH_PYTHON' ''
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
    $launcher = Get-Command 'blindassist-python.cmd' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($launcher) { return $launcher.Source }
    $pythonCommand = Get-Command 'python' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pythonCommand) { return $pythonCommand.Source }
    return $candidate
}

function Resolve-ExportPython {
    $local = Read-LocalConfig
    $value = if ($local.ContainsKey('export_python')) { $local['export_python'] } else { '' }
    if ([string]::IsNullOrWhiteSpace($value)) { $value = $env:BLINDASSIST_EXPORT_PYTHON }
    if ([string]::IsNullOrWhiteSpace($value)) { $value = '.venv-export/Scripts/python.exe' }
    return Resolve-ConfiguredPath $value '__unused__' '__UNUSED__' $value
}

function Write-Selection {
    param([string]$ResearchPython, [string]$OutputPath)
    Write-Host "repo: $RepoRoot"
    if ($ResearchPython) { Write-Host "python: $ResearchPython" }
    if ($OutputPath) { Write-Host "output: $OutputPath" }
}

function Get-ResearchSelection {
    $selectedPython = Resolve-ResearchPython
    $selectedOutput = Resolve-ConfiguredPath '' 'dtr_r0_output' 'BLINDASSIST_DTR_R0_OUTPUT' 'artifacts.local/evidence/dtr-r0'
    return @{ Python = $selectedPython; Output = $selectedOutput }
}

function Invoke-DoctorBase {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot '.git'))) {
        Stop-Ba 'BA_ENV_REPO_INVALID' "Git metadata is missing below $RepoRoot" 'open the command from a BlindAssist checkout or worktree'
    }
    Write-Output 'PASS base'
    Write-Output "repo: $RepoRoot"
    Write-Output "powershell: $($PSVersionTable.PSVersion)"
    Write-Output "git: $(& git --version)"
}

function Invoke-DoctorResearch {
    $selection = Get-ResearchSelection
    Write-Selection $selection.Python $selection.Output
    if (-not $selection.Python -or -not (Test-Path -LiteralPath $selection.Python -PathType Leaf)) {
        Stop-Ba 'BA_ENV_PYTHON_MISSING' "research Python is unavailable: $($selection.Python)" 'install Python 3.11+ or set research_python in config/local.toml'
    }
    foreach ($required in @('README.md', 'dtr_r0.py', 'evaluate.py', 'generate_smoke.py', 'test_dtr_r0.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $ActiveRoot $required) -PathType Leaf)) {
            Stop-Ba 'BA_ENV_ACTIVE_ROUTE_INCOMPLETE' "missing $required below $ActiveRoot" 'restore the tracked DTR-R0 active route'
        }
    }
    $probe = 'import json,platform,sys; print(json.dumps(dict(python=platform.python_version(),supported=sys.version_info >= (3, 11))))'
    $probeOutput = @(& $selection.Python -c $probe 2>&1)
    if ($LASTEXITCODE -ne 0) {
        Stop-Ba 'BA_ENV_RESEARCH_IMPORT_FAILED' ($probeOutput -join ' ') 'install Python 3.11+ or set research_python in config/local.toml'
    }
    $runtimeLine = $probeOutput | ForEach-Object { $_.ToString() } | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
    if (-not $runtimeLine) {
        Stop-Ba 'BA_ENV_RESEARCH_PROBE_INVALID' ($probeOutput -join ' ') 'install Python 3.11+ or set research_python in config/local.toml'
    }
    $runtime = $runtimeLine | ConvertFrom-Json
    Write-Host "runtime: $($runtime | ConvertTo-Json -Compress)"
    if (-not $runtime.supported) {
        Stop-Ba 'BA_ENV_PYTHON_UNSUPPORTED' "Python $($runtime.python) is older than 3.11" 'install Python 3.11+ or set research_python in config/local.toml'
    }
    Write-Host 'PASS research-dtr-r0'
    return $selection
}

function Invoke-DoctorAndroid {
    $gradle = Join-Path $RepoRoot 'scripts/run_android_gradle.ps1'
    & $gradle -PreflightOnly -RequireDevice:($Profile -eq 'device')
    if ($LASTEXITCODE -ne 0) {
        $code = if ($Profile -eq 'device') { 'BA_ENV_DEVICE_UNAVAILABLE' } else { 'BA_ENV_ANDROID_UNAVAILABLE' }
        Stop-Ba $code 'Android preflight failed' "pwsh -NoProfile -File tools/ba.ps1 doctor $Profile"
    }
    Write-Output "PASS $Profile"
}

function Invoke-DoctorExport {
    $selected = Resolve-ExportPython
    Write-Output "repo: $RepoRoot"
    Write-Output "python: $selected"
    if (-not (Test-Path -LiteralPath $selected -PathType Leaf)) {
        Stop-Ba 'BA_ENV_EXPORT_PYTHON_MISSING' "export Python is unavailable: $selected" 'pwsh -NoProfile -File tools/ba.ps1 setup export'
    }
    & $selected -c 'import ai_edge_litert, onnx, onnxruntime, tensorflow, ultralytics; print("PASS export imports")'
    if ($LASTEXITCODE -ne 0) {
        Stop-Ba 'BA_ENV_EXPORT_IMPORT_FAILED' 'one or more export packages are unavailable' 'pwsh -NoProfile -File tools/ba.ps1 setup export'
    }
}

function Invoke-Setup {
    switch ($Profile) {
        'base' { Invoke-DoctorBase }
        'research-dtr-r0' { Invoke-DoctorResearch | Out-Null }
        'android' { Invoke-DoctorAndroid }
        'device' { Invoke-DoctorAndroid }
        'export' {
            $uv = Get-Command uv -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $uv) { Stop-Ba 'BA_ENV_UV_MISSING' 'uv was not found' 'install uv, then rerun setup export' }
            $venv = Join-Path $RepoRoot '.venv-export'
            if (-not (Test-Path -LiteralPath (Join-Path $venv 'Scripts/python.exe'))) {
                Invoke-NativeChecked $uv.Source @('venv', $venv, '--python', '3.12')
            }
            Invoke-NativeChecked $uv.Source @('pip', 'install', '--python', (Join-Path $venv 'Scripts/python.exe'), '-r', (Join-Path $RepoRoot 'requirements-export.txt'))
            Invoke-DoctorExport
        }
    }
}

function Invoke-Doctor {
    switch ($Profile) {
        'base' { Invoke-DoctorBase }
        'research-dtr-r0' { Invoke-DoctorResearch | Out-Null }
        'android' { Invoke-DoctorAndroid }
        'device' { Invoke-DoctorAndroid }
        'export' { Invoke-DoctorExport }
    }
}

function Invoke-Smoke {
    if ($Profile -ne 'research-dtr-r0') { Stop-Ba 'BA_USAGE' 'smoke is currently defined only for research-dtr-r0' 'use tools/ba.ps1 doctor for this profile' }
    $selection = Invoke-DoctorResearch
    Invoke-NativeChecked $selection.Python @('-m', 'unittest', 'discover', '-s', $ActiveRoot, '-p', 'test_*.py')
}

function Invoke-Run {
    $forward = @($Arguments | Where-Object { $_ -ne '--' })
    switch ($Profile) {
        'research-dtr-r0' {
            $selection = Invoke-DoctorResearch
            if ([string]::IsNullOrWhiteSpace($EventInput)) {
                Stop-Ba 'BA_USAGE' 'DTR-R0 run needs -EventInput' 'tools/ba.ps1 run research-dtr-r0 -EventInput <events.jsonl> -ResultOutput <result.json>'
            }
            $resolvedInput = Resolve-ConfiguredPath $EventInput '__unused__' '__UNUSED__' ''
            $evaluationArguments = @((Join-Path $ActiveRoot 'evaluate.py'), '--input', $resolvedInput)
            if (-not [string]::IsNullOrWhiteSpace($ResultOutput)) {
                $resolvedOutput = Resolve-ConfiguredPath $ResultOutput '__unused__' '__UNUSED__' ''
                $evaluationArguments += @('--output', $resolvedOutput)
            }
            Invoke-NativeChecked $selection.Python $evaluationArguments
        }
        'android' { & (Join-Path $RepoRoot 'scripts/run_android_gradle.ps1') @forward; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
        'device' { & (Join-Path $RepoRoot 'scripts/run_android_gradle.ps1') -RequireDevice @forward; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
        'export' {
            Invoke-DoctorExport
            if ($forward.Count -lt 1) { Stop-Ba 'BA_USAGE' 'export run needs a script and its arguments' 'tools/ba.ps1 run export -- <script> <arguments>' }
            $selected = Resolve-ExportPython
            Invoke-NativeChecked $selected $forward
        }
        default { Stop-Ba 'BA_USAGE' 'base has no run target' 'choose research-dtr-r0, android, device, or export' }
    }
}

function Invoke-Clean {
    $target = switch ($Profile) {
        'export' { Join-Path $RepoRoot '.venv-export' }
        default { $null }
    }
    if (-not $target) {
        Stop-Ba 'BA_CLEAN_REFUSED' "$Profile has no profile-owned disposable environment" 'clean only export'
    }
    $resolvedParent = (Resolve-Path -LiteralPath (Split-Path -Parent $target)).Path
    if (-not $target.StartsWith($resolvedParent + '\', [StringComparison]::OrdinalIgnoreCase)) {
        Stop-Ba 'BA_CLEAN_REFUSED' "target escaped its owned parent: $target" 'inspect the profile configuration'
    }
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    if (Test-Path -LiteralPath $target) { Stop-Ba 'BA_CLEAN_FAILED' "target remains: $target" 'close processes using the profile environment and retry' }
    Write-Output "PASS cleaned $target"
}

try {
    switch ($Command) {
        'setup' { Invoke-Setup }
        'doctor' { Invoke-Doctor }
        'smoke' { Invoke-Smoke }
        'run' { Invoke-Run }
        'clean' { Invoke-Clean }
    }
} catch {
    Stop-Ba 'BA_INTERNAL_ERROR' $_.Exception.Message 'fix the first reported project entrypoint error and rerun the same command'
}
