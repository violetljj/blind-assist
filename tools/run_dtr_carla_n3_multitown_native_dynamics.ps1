[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [string]$Registry = 'research/active/dtr-r0/carla/dtr_carla_n3_multitown_native_dynamics_registry.json',
    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$EvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-n3-multitown-native-dynamics\evidence',
    [string]$Compiler = 'research/active/dtr-r0/carla/dtr_carla_n3_multitown_native_dynamics.py',
    [string]$Sealer = 'research/active/dtr-r0/carla/seal_dtr_carla_n3_multitown_native_dynamics.py',
    [string]$N1Runner = 'tools/run_dtr_carla_n1_natural_dynamics.ps1',
    [ValidateRange(1024, 65529)]
    [int]$RpcPort = 26200,
    [ValidateRange(1024, 65535)]
    [int]$TrafficManagerPort = 26203,
    [ValidateRange(120, 3600)]
    [int]$CaptureTimeoutSeconds = 1200,
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

function Get-EngineMapObjectPath {
    param([Parameter(Mandatory = $true)][string]$MapName)
    $normalized = $MapName.TrimStart('/')
    if ($normalized -notmatch '^Carla/Maps/(?<leaf>[A-Za-z0-9_]+)$') {
        throw "Unsupported CARLA map identity: $MapName"
    }
    return "/Game/$normalized.$($Matches.leaf)"
}

function New-StartupEngineIni {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$MapName
    )
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing startup Engine.ini overwrite: $Path"
    }
    [IO.Directory]::CreateDirectory((Split-Path -Parent $Path)) | Out-Null
    $objectPath = Get-EngineMapObjectPath -MapName $MapName
    $gameMode = '/Game/Carla/Blueprints/Game/CarlaGameMode.CarlaGameMode_C'
    $content = @(
        '[/Script/EngineSettings.GameMapsSettings]',
        "GameDefaultMap=$objectPath",
        "ServerDefaultMap=$objectPath",
        "TransitionMap=$objectPath",
        "GlobalDefaultGameMode=$gameMode",
        "GlobalDefaultServerGameMode=$gameMode",
        ''
    ) -join "`n"
    [IO.File]::WriteAllText($Path, $content, [Text.UTF8Encoding]::new($false))
}

$registryPath = Resolve-TaskPath $Registry
$compilerPath = Resolve-TaskPath $Compiler
$sealerPath = Resolve-TaskPath $Sealer
$n1RunnerPath = Resolve-TaskPath $N1Runner
$carlaPythonPath = Resolve-TaskPath $CarlaPython
foreach ($required in @(
    $registryPath,
    $compilerPath,
    $sealerPath,
    $n1RunnerPath,
    $carlaPythonPath
)) {
    Assert-RequiredFile -Path $required
}

$evidenceRootPath = [IO.Path]::GetFullPath($EvidenceRoot).TrimEnd('\', '/')
$runRoot = [IO.Path]::GetFullPath((Join-Path $evidenceRootPath $RunId))
$expectedPrefix = $evidenceRootPath + [IO.Path]::DirectorySeparatorChar
if (-not $runRoot.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Run path escapes N3 evidence root: $runRoot"
}
if (Test-Path -LiteralPath $runRoot) {
    throw "Refusing N3 evidence overwrite: $runRoot"
}
[IO.Directory]::CreateDirectory($runRoot) | Out-Null

$frozenInputs = Join-Path $runRoot 'frozen-inputs'
$compileOutput = @(
    & $carlaPythonPath -B $compilerPath `
        --registry $registryPath `
        --output-root $frozenInputs 2>&1
)
if ($LASTEXITCODE -ne 0) {
    throw "N3 suite compile failed: $($compileOutput -join [Environment]::NewLine)"
}
$suiteManifestPath = Join-Path $frozenInputs 'suite_manifest.json'
$suite = Get-Content -LiteralPath $suiteManifestPath -Raw | ConvertFrom-Json -Depth 100
if (
    [string]$suite.schema_version -ne 'dtr-carla-n3-multitown-native-dynamics-suite-v1' -or
    [int]$suite.scene_count -ne 3
) {
    throw 'N3 compiled suite identity differs.'
}

$startupRoot = Join-Path $frozenInputs 'startup-engine'
$tracesRoot = Join-Path $runRoot 'source-traces'
foreach ($scene in @($suite.scenes)) {
    $sceneId = [string]$scene.scene_id
    $planPath = Join-Path $frozenInputs ([string]$scene.plan_path)
    Assert-RequiredFile -Path $planPath
    $engineIni = Join-Path $startupRoot "$sceneId.Engine.ini"
    New-StartupEngineIni -Path $engineIni -MapName ([string]$scene.map)
    Write-Output (
        "START N3 native trace scene=$sceneId map=$($scene.map) " +
        "class=$($scene.scenario_class)"
    )
    & pwsh -NoProfile -File $n1RunnerPath `
        -RunId $sceneId `
        -Plan $planPath `
        -CarlaRoot $CarlaRoot `
        -CarlaPython $carlaPythonPath `
        -RawEvidenceRoot $tracesRoot `
        -StartupEngineIni $engineIni `
        -RpcPort $RpcPort `
        -TrafficManagerPort $TrafficManagerPort `
        -CaptureTimeoutSeconds $CaptureTimeoutSeconds `
        -MinimumFreePhysicalGB $MinimumFreePhysicalGB
    if ($LASTEXITCODE -ne 0) {
        throw "N3 native trace failed for $sceneId with exit code $LASTEXITCODE"
    }
    $childResultPath = Join-Path $tracesRoot "$sceneId\result.json"
    Assert-RequiredFile -Path $childResultPath
    $childResult = Get-Content -LiteralPath $childResultPath -Raw | ConvertFrom-Json -Depth 100
    $failedChecks = @(
        $childResult.checks.PSObject.Properties |
            Where-Object { $_.Value -isnot [bool] -or $_.Value -ne $true }
    )
    if (
        [string]$childResult.status -ne 'DTR_CARLA_N1_NATURAL_DYNAMICS_MATERIALIZED' -or
        $failedChecks.Count -ne 0
    ) {
        throw "N3 child gate failed for ${sceneId}: $($failedChecks.Name -join ',')"
    }
    Write-Output "PASS N3 native trace $sceneId"
    if ($CooldownSeconds -gt 0 -and $sceneId -ne [string]$suite.scenes[-1].scene_id) {
        Start-Sleep -Seconds $CooldownSeconds
    }
}

$sealOutput = @(
    & $carlaPythonPath -B $sealerPath `
        --suite-manifest $suiteManifestPath `
        --traces-root $tracesRoot `
        --output-root $runRoot 2>&1
)
if ($LASTEXITCODE -ne 0) {
    throw "N3 suite seal failed: $($sealOutput -join [Environment]::NewLine)"
}
$resultPath = Join-Path $runRoot 'result.json'
$result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json -Depth 100
if ([string]$result.status -ne 'DTR_CARLA_N3_MULTITOWN_NATIVE_DYNAMICS_MATERIALIZED') {
    throw "N3 suite result gate failed: $($result.status)"
}
Write-Output "PASS N3 multi-town native dynamics: $resultPath"
