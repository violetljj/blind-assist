[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'doctor', 'smoke', 'run', 'clean')]
    [string]$Command = 'doctor',
    [Parameter(Position = 1)]
    [ValidateSet('base', 'research-r1cl', 'android', 'device', 'export')]
    [string]$Profile = 'base',
    [string]$Python,
    [string]$Backbone,
    [string]$TrainDataset,
    [string]$ValidationDataset,
    [string]$Docker,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$ActiveRoot = Join-Path $RepoRoot 'research/active/grail-r1cl'
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
    $candidate = Resolve-ConfiguredPath $Python 'research_python' 'BLINDASSIST_RESEARCH_PYTHON' 'research/active/grail-r1cl/.venv/Scripts/python.exe'
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
    if ([string]::IsNullOrWhiteSpace($value)) { $value = 'research/active/grail-r1cl/.venv/Scripts/python.exe' }
    return Resolve-ConfiguredPath $value '__unused__' '__UNUSED__' $value
}

function Write-Selection {
    param([string]$ResearchPython, [string]$BackbonePath, [string]$TrainPath, [string]$ValidationPath, [string]$OutputPath)
    Write-Host "repo: $RepoRoot"
    if ($ResearchPython) { Write-Host "python: $ResearchPython" }
    if ($BackbonePath) { Write-Host "backbone: $BackbonePath" }
    if ($TrainPath) { Write-Host "train_dataset: $TrainPath" }
    if ($ValidationPath) { Write-Host "validation_dataset: $ValidationPath" }
    if ($OutputPath) { Write-Host "output: $OutputPath" }
}

function Get-ResearchSelection {
    $selectedPython = Resolve-ResearchPython
    $selectedBackbone = Resolve-ConfiguredPath $Backbone 'r1cl_backbone' 'BLINDASSIST_R1CL_BACKBONE' 'artifacts.local/models/dinov2-small'
    $selectedTrain = Resolve-ConfiguredPath $TrainDataset 'r1cl_train_dataset' 'BLINDASSIST_R1CL_TRAIN_DATASET' 'artifacts.local/datasets/grail-r1cl/train.jsonl.gz'
    $selectedValidation = Resolve-ConfiguredPath $ValidationDataset 'r1cl_validation_dataset' 'BLINDASSIST_R1CL_VALIDATION_DATASET' ''
    $selectedOutput = Resolve-ConfiguredPath '' 'r1cl_output' 'BLINDASSIST_R1CL_OUTPUT' 'artifacts.local/evidence/grail-r1cl'
    return @{
        Python = $selectedPython; Backbone = $selectedBackbone; Train = $selectedTrain
        Validation = $selectedValidation; Output = $selectedOutput
    }
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
    Write-Selection $selection.Python $selection.Backbone $selection.Train $selection.Validation $selection.Output
    if (-not $selection.Python -or -not (Test-Path -LiteralPath $selection.Python -PathType Leaf)) {
        Stop-Ba 'BA_ENV_PYTHON_MISSING' "research Python is unavailable: $($selection.Python)" 'pwsh -NoProfile -File tools/ba.ps1 setup research-r1cl'
    }
    $probe = 'import json,platform,numpy,PIL,torch,transformers; print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,pillow=PIL.__version__,torch=torch.__version__,transformers=transformers.__version__,cuda_available=torch.cuda.is_available(),torch_cuda=torch.version.cuda,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))'
    $probeOutput = @(& $selection.Python -c $probe 2>&1)
    if ($LASTEXITCODE -ne 0) {
        Stop-Ba 'BA_ENV_RESEARCH_IMPORT_FAILED' ($probeOutput -join ' ') 'pwsh -NoProfile -File tools/ba.ps1 setup research-r1cl'
    }
    $runtimeLine = $probeOutput | ForEach-Object { $_.ToString() } | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
    if (-not $runtimeLine) {
        Stop-Ba 'BA_ENV_RESEARCH_PROBE_INVALID' ($probeOutput -join ' ') 'pwsh -NoProfile -File tools/ba.ps1 setup research-r1cl'
    }
    $runtime = $runtimeLine | ConvertFrom-Json
    Write-Host "runtime: $($runtime | ConvertTo-Json -Compress)"
    if (-not $runtime.cuda_available) {
        Stop-Ba 'BA_ENV_CUDA_UNAVAILABLE' "torch=$($runtime.torch) torch_cuda=$($runtime.torch_cuda) gpu_visible=false" 'install/activate the CUDA research runtime, then run tools/ba.ps1 doctor research-r1cl'
    }
    foreach ($required in @('config.json', 'model.safetensors')) {
        if (-not (Test-Path -LiteralPath (Join-Path $selection.Backbone $required) -PathType Leaf)) {
            Stop-Ba 'BA_ENV_BACKBONE_MISSING' "missing $required below $($selection.Backbone)" 'set r1cl_backbone in config/local.toml, then rerun doctor'
        }
    }
    if (-not (Test-Path -LiteralPath $selection.Train -PathType Leaf)) {
        Stop-Ba 'BA_ENV_TRAIN_DATA_MISSING' "training dataset is unavailable: $($selection.Train)" 'set r1cl_train_dataset in config/local.toml, then rerun doctor'
    }
    Write-Host 'PASS research-r1cl'
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
        'research-r1cl' {
            $selection = Get-ResearchSelection
            if (-not $selection.Python -or -not (Test-Path -LiteralPath $selection.Python -PathType Leaf)) {
                $uv = Get-Command uv -ErrorAction SilentlyContinue | Select-Object -First 1
                if (-not $uv) { Stop-Ba 'BA_ENV_UV_MISSING' 'uv was not found' 'install uv or set research_python in config/local.toml' }
                Push-Location $ActiveRoot
                try { Invoke-NativeChecked $uv.Source @('sync', '--frozen') } finally { Pop-Location }
            }
            Invoke-DoctorResearch | Out-Null
        }
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
        'research-r1cl' { Invoke-DoctorResearch | Out-Null }
        'android' { Invoke-DoctorAndroid }
        'device' { Invoke-DoctorAndroid }
        'export' { Invoke-DoctorExport }
    }
}

function Invoke-Smoke {
    if ($Profile -ne 'research-r1cl') { Stop-Ba 'BA_USAGE' 'smoke is currently defined only for research-r1cl' 'use tools/ba.ps1 doctor for this profile' }
    $selection = Invoke-DoctorResearch
    Invoke-NativeChecked $selection.Python @((Join-Path $ActiveRoot 'smoke_r1cl.py'), '--backbone', $selection.Backbone)
}

function Invoke-Run {
    $forward = @($Arguments | Where-Object { $_ -ne '--' })
    switch ($Profile) {
        'research-r1cl' {
            $selection = Invoke-DoctorResearch
            $trainingArguments = @((Join-Path $ActiveRoot 'train_grail_pairwise_owner_coordinate_r1cl.py')) + $forward
            Invoke-NativeChecked $selection.Python $trainingArguments
        }
        'android' { & (Join-Path $RepoRoot 'scripts/run_android_gradle.ps1') @forward; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
        'device' { & (Join-Path $RepoRoot 'scripts/run_android_gradle.ps1') -RequireDevice @forward; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
        'export' {
            Invoke-DoctorExport
            if ($forward.Count -lt 1) { Stop-Ba 'BA_USAGE' 'export run needs a script and its arguments' 'tools/ba.ps1 run export -- <script> <arguments>' }
            $selected = Resolve-ExportPython
            Invoke-NativeChecked $selected $forward
        }
        default { Stop-Ba 'BA_USAGE' 'base has no run target' 'choose research-r1cl, android, device, or export' }
    }
}

function Invoke-Clean {
    $target = switch ($Profile) {
        'research-r1cl' { Join-Path $ActiveRoot '.venv' }
        'export' { Join-Path $RepoRoot '.venv-export' }
        default { $null }
    }
    if (-not $target) { Stop-Ba 'BA_CLEAN_REFUSED' "$Profile has no profile-owned disposable environment" 'clean only research-r1cl or export' }
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
