[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [string]$SourceRunRoot,

    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$EvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-n4-multitown-frozen-replay\evidence',
    [string]$Builder = 'research/active/dtr-r0/carla/build_dtr_carla_n4_multitown_frozen_replay.py',
    [string]$Joiner = 'research/active/dtr-r0/carla/join_dtr_carla_n4_multitown_frozen_replay.py',
    [string]$N2Runner = 'tools/run_dtr_carla_n2_frozen_trace_replay.ps1',
    [ValidateRange(1024, 65533)]
    [int]$RpcPort = 26300,
    [ValidateRange(120, 3600)]
    [int]$CaptureTimeoutSeconds = 2400,
    [ValidateRange(2.0, 16.0)]
    [double]$MinimumFreePhysicalGB = 4.0,
    [ValidateRange(0, 120)]
    [int]$CooldownSeconds = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

function Resolve-TaskPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

function Assert-RequiredFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is unavailable: $Path"
    }
}

function New-StartupEngineIni {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$EngineMapObjectPath
    )
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing startup Engine.ini overwrite: $Path"
    }
    [IO.Directory]::CreateDirectory((Split-Path -Parent $Path)) | Out-Null
    $gameMode = '/Game/Carla/Blueprints/Game/CarlaGameMode.CarlaGameMode_C'
    $content = @(
        '[/Script/EngineSettings.GameMapsSettings]',
        "GameDefaultMap=$EngineMapObjectPath",
        "ServerDefaultMap=$EngineMapObjectPath",
        "TransitionMap=$EngineMapObjectPath",
        "GlobalDefaultGameMode=$gameMode",
        "GlobalDefaultServerGameMode=$gameMode",
        ''
    ) -join "`n"
    [IO.File]::WriteAllText($Path, $content, [Text.UTF8Encoding]::new($false))
}

function Write-JsonExclusive {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing JSON overwrite: $Path"
    }
    $text = ($Value | ConvertTo-Json -Depth 100) + [Environment]::NewLine
    [IO.File]::WriteAllText($Path, $text, [Text.UTF8Encoding]::new($false))
}

$sourceRunPath = Resolve-TaskPath $SourceRunRoot
$builderPath = Resolve-TaskPath $Builder
$joinerPath = Resolve-TaskPath $Joiner
$n2RunnerPath = Resolve-TaskPath $N2Runner
$carlaPythonPath = Resolve-TaskPath $CarlaPython
foreach ($required in @($builderPath, $joinerPath, $n2RunnerPath, $carlaPythonPath)) {
    Assert-RequiredFile -Path $required
}
if (-not (Test-Path -LiteralPath $sourceRunPath -PathType Container)) {
    throw "N3 source run is unavailable: $sourceRunPath"
}
$sourceResultPath = Join-Path $sourceRunPath 'result.json'
Assert-RequiredFile -Path $sourceResultPath
$sourceResult = Get-Content -LiteralPath $sourceResultPath -Raw | ConvertFrom-Json -Depth 100
if ([string]$sourceResult.status -ne 'DTR_CARLA_N3_MULTITOWN_NATIVE_DYNAMICS_MATERIALIZED') {
    throw "N3 source run is not complete: $($sourceResult.status)"
}

$evidenceRootPath = [IO.Path]::GetFullPath($EvidenceRoot).TrimEnd('\', '/')
$runRoot = [IO.Path]::GetFullPath((Join-Path $evidenceRootPath $RunId))
$expectedPrefix = $evidenceRootPath + [IO.Path]::DirectorySeparatorChar
if (-not $runRoot.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Run path escapes N4 evidence root: $runRoot"
}
if (Test-Path -LiteralPath $runRoot) {
    throw "Refusing N4 evidence overwrite: $runRoot"
}
[IO.Directory]::CreateDirectory($runRoot) | Out-Null

$frozenInputs = Join-Path $runRoot 'frozen-inputs'
$buildOutput = @(
    & $carlaPythonPath -B $builderPath `
        --source-run-root $sourceRunPath `
        --output-root $frozenInputs 2>&1
)
if ($LASTEXITCODE -ne 0) {
    throw "N4 frozen replay bundle build failed: $($buildOutput -join [Environment]::NewLine)"
}
$bundlePath = Join-Path $frozenInputs 'bundle.json'
$bundle = Get-Content -LiteralPath $bundlePath -Raw | ConvertFrom-Json -Depth 100
if (
    [string]$bundle.schema_version -ne 'dtr-carla-n4-multitown-frozen-replay-bundle-v1' -or
    [int]$bundle.replay_invocation_budget -ne 1 -or
    [int]$bundle.scene_count -ne 3
) {
    throw 'N4 frozen replay bundle identity differs.'
}

$attemptReceiptPath = Join-Path $runRoot 'replay_attempt_receipt.json'
Write-JsonExclusive -Path $attemptReceiptPath -Value ([ordered]@{
    schema_version = 'dtr-carla-n4-replay-attempt-receipt-v1'
    authority = 'SOLE_FROZEN_FOUR_MODAL_REPLAY_ATTEMPT_CONSUMED'
    replay_invocation_count = 1
    bundle_sha256 = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash
    source_result_sha256 = (Get-FileHash -LiteralPath $sourceResultPath -Algorithm SHA256).Hash
    run_id = $RunId
    map_shards = @($bundle.children | ForEach-Object { [string]$_.scene_id })
})

$startupRoot = Join-Path $frozenInputs 'startup-engine'
$childRoot = Join-Path $runRoot 'child-evidence'
foreach ($child in @($bundle.children)) {
    $sceneId = [string]$child.scene_id
    $protocolPath = Join-Path $frozenInputs ([string]$child.protocol_path)
    $sourceRoot = [IO.Path]::GetFullPath([string]$child.source_root)
    Assert-RequiredFile -Path $protocolPath
    if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        throw "N4 frozen source root is unavailable: $sourceRoot"
    }
    if ((Get-FileHash -LiteralPath $protocolPath -Algorithm SHA256).Hash -ne [string]$child.protocol_sha256) {
        throw "N4 child protocol hash differs before replay: $sceneId"
    }
    $engineIni = Join-Path $startupRoot "$sceneId.Engine.ini"
    New-StartupEngineIni `
        -Path $engineIni `
        -EngineMapObjectPath ([string]$child.engine_map_object_path)
    Write-Output (
        "START sole N4 replay shard scene=$sceneId map=$($child.map) " +
        'modalities=instance,wearable,depth,witness'
    )
    & pwsh -NoProfile -File $n2RunnerPath `
        -RunId $sceneId `
        -SourceRoot $sourceRoot `
        -CarlaRoot $CarlaRoot `
        -CarlaPython $carlaPythonPath `
        -EvidenceRoot $childRoot `
        -Protocol $protocolPath `
        -StartupEngineIni $engineIni `
        -RpcPort $RpcPort `
        -CaptureTimeoutSeconds $CaptureTimeoutSeconds `
        -MinimumFreePhysicalGB $MinimumFreePhysicalGB
    if ($LASTEXITCODE -ne 0) {
        throw "N4 replay shard failed for $sceneId with exit code $LASTEXITCODE; replay is consumed and must not be retried."
    }
    $childResultPath = Join-Path $childRoot "$sceneId\result.json"
    Assert-RequiredFile -Path $childResultPath
    $childResult = Get-Content -LiteralPath $childResultPath -Raw | ConvertFrom-Json -Depth 100
    $failedChecks = @(
        $childResult.checks.PSObject.Properties |
            Where-Object { $_.Value -isnot [bool] -or $_.Value -ne $true }
    )
    if (
        [string]$childResult.status -ne 'DTR_CARLA_N2_FROZEN_TRACE_C2_REPLAY_COMPLETE' -or
        $failedChecks.Count -ne 0
    ) {
        throw "N4 replay shard gate failed for ${sceneId}: $($failedChecks.Name -join ','); replay is consumed and must not be retried."
    }
    Write-Output "PASS N4 replay shard $sceneId"
    if ($CooldownSeconds -gt 0 -and $sceneId -ne [string]$bundle.children[-1].scene_id) {
        Start-Sleep -Seconds $CooldownSeconds
    }
}

$joinOutput = @(
    & $carlaPythonPath -B $joinerPath `
        --bundle $bundlePath `
        --child-root $childRoot `
        --output-root $runRoot 2>&1
)
if ($LASTEXITCODE -ne 0) {
    throw "N4 final join failed: $($joinOutput -join [Environment]::NewLine)"
}
$resultPath = Join-Path $runRoot 'result.json'
$result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json -Depth 100
if ([string]$result.status -ne 'DTR_CARLA_N4_MULTITOWN_FROZEN_FOUR_MODAL_REPLAY_COMPLETE') {
    throw "N4 final result gate failed: $($result.status)"
}
Write-Output "PASS sole N4 multi-town frozen four-modal replay: $resultPath"
