[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [string]$SceneId = 'town10_dense_risk_canary',
    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$RawEvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-c3-dynamic-risk\evidence',
    [ValidateRange(1024, 65533)]
    [int]$RpcPort = 2000,
    [string]$AssetRegistry = 'research/active/dtr-r0/carla/dtr_carla_c3_asset_registry.json',
    [string]$SceneRegistry = 'research/active/dtr-r0/carla/dtr_carla_c3_scene_registry.json',
    [ValidateRange(120, 7200)]
    [int]$CaptureTimeoutSeconds = 3600
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

function Resolve-TaskPath {
    param([string]$Value)
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

function Assert-File {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is unavailable: $Path"
    }
}

function Invoke-NativeChecked {
    param([scriptblock]$Command, [string]$Label)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$baseProtocol = Join-Path $repoRoot 'research/active/dtr-r0/carla/dtr_carla_c2_rich_scene_protocol.json'
$compiler = Join-Path $repoRoot 'research/active/dtr-r0/carla/compile_dtr_carla_c3_scene.py'
$c3Join = Join-Path $repoRoot 'research/active/dtr-r0/carla/join_dtr_carla_c3_dynamic_risk.py'
$c2Runner = Join-Path $repoRoot 'tools/run_dtr_carla_c2_rich_scene.ps1'
$assetPath = Resolve-TaskPath -Value $AssetRegistry
$scenePath = Resolve-TaskPath -Value $SceneRegistry
$pythonPath = Resolve-TaskPath -Value $CarlaPython
$evidenceRoot = Resolve-TaskPath -Value $RawEvidenceRoot
$evidencePath = [IO.Path]::GetFullPath((Join-Path $evidenceRoot $RunId))
$compileBase = Join-Path $repoRoot 'artifacts.local/carla-c3-compiled'
$compilePath = [IO.Path]::GetFullPath((Join-Path $compileBase $RunId))
$compilePrefix = [IO.Path]::GetFullPath($compileBase).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if (-not $compilePath.StartsWith($compilePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Compile path escapes task-owned root: $compilePath"
}
$compiledProtocol = Join-Path $compilePath 'compiled-c2-protocol.json'
$compilerReceipt = Join-Path $compilePath 'compiler-receipt.json'
$frozenAssetRegistry = Join-Path $compilePath 'asset-registry.json'
$frozenSceneRegistry = Join-Path $compilePath 'scene-registry.json'
$c2RunnerCommand = Get-Command -Name $c2Runner -ErrorAction Stop
$c2RunnerSupportsRpcPort = $c2RunnerCommand.Parameters.ContainsKey('RpcPort')
if (-not $c2RunnerSupportsRpcPort -and $RpcPort -ne 2000) {
    throw 'The checked-out C2 compatibility runner does not support an isolated RPC port.'
}

foreach ($required in @(
    @($baseProtocol, 'C2 base protocol'),
    @($compiler, 'C3 compiler'),
    @($c3Join, 'C3 join'),
    @($c2Runner, 'C2 compatibility runner'),
    @($assetPath, 'C3 asset registry'),
    @($scenePath, 'C3 scene registry'),
    @($pythonPath, 'CARLA Python')
)) {
    Assert-File -Path $required[0] -Label $required[1]
}
if (Test-Path -LiteralPath $compilePath) {
    throw "Refusing compile workspace overwrite: $compilePath"
}
if (Test-Path -LiteralPath $evidencePath) {
    throw "Refusing evidence overwrite: $evidencePath"
}

$complete = $false
try {
    [IO.Directory]::CreateDirectory($compilePath) | Out-Null
    Copy-Item -LiteralPath $assetPath -Destination $frozenAssetRegistry
    Copy-Item -LiteralPath $scenePath -Destination $frozenSceneRegistry
    if (
        (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $frozenAssetRegistry -Algorithm SHA256).Hash -or
        (Get-FileHash -LiteralPath $scenePath -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $frozenSceneRegistry -Algorithm SHA256).Hash
    ) {
        throw 'Frozen C3 registry copies differ from their source files.'
    }
    Push-Location $repoRoot
    try {
        Invoke-NativeChecked -Label 'C3 registry compilation' -Command {
            & $pythonPath $compiler `
                --base-c2-protocol $baseProtocol `
                --asset-registry $frozenAssetRegistry `
                --scene-registry $frozenSceneRegistry `
                --scene-id $SceneId `
                --output-protocol $compiledProtocol `
                --output-receipt $compilerReceipt
        }
        Invoke-NativeChecked -Label 'C3 C2-compatible 1280x720 capture/join' -Command {
            $runnerArguments = @(
                '-NoProfile', '-File', $c2Runner,
                '-RunId', $RunId,
                '-CarlaRoot', $CarlaRoot,
                '-CarlaPython', $pythonPath,
                '-RawEvidenceRoot', $evidenceRoot,
                '-Protocol', $compiledProtocol,
                '-CaptureTimeoutSeconds', [string]$CaptureTimeoutSeconds
            )
            if ($c2RunnerSupportsRpcPort) {
                $runnerArguments += @('-RpcPort', [string]$RpcPort)
            }
            & pwsh @runnerArguments
        }
        Invoke-NativeChecked -Label 'C3 dynamic-risk registry join' -Command {
            & $pythonPath $c3Join `
                --root $evidencePath `
                --asset-registry $frozenAssetRegistry `
                --scene-registry $frozenSceneRegistry `
                --compiler-receipt $compilerReceipt `
                --scene-id $SceneId
        }
    }
    finally {
        Pop-Location
    }
    $resultPath = Join-Path $evidencePath 'result.json'
    Assert-File -Path $resultPath -Label 'C3 result'
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json -Depth 100
    $failed = @($result.checks.PSObject.Properties | Where-Object { -not [bool]$_.Value })
    if (
        [string]$result.status -ne 'DTR_CARLA_C3_DENSE_DYNAMIC_RISK_SOURCE_COMPLETE' -or
        $failed.Count -ne 0
    ) {
        throw "C3 final gate failed: status=$($result.status) checks=$($failed.Name -join ',')"
    }
    $complete = $true
    Write-Output (
        'PASS DTR-CARLA-C3 dynamic-risk source: actors=40 dynamic_risk=16 ' +
        'resolution=1280x720 sensors=4'
    )
    Write-Output "evidence: $evidencePath"
}
finally {
    if ($complete -and (Test-Path -LiteralPath $compilePath -PathType Container)) {
        Remove-Item -LiteralPath $compilePath -Recurse -Force
        if (Test-Path -LiteralPath $compilePath) {
            throw "Task-owned compile workspace remained after cleanup: $compilePath"
        }
    }
}
