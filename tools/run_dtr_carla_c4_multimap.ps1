[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [string]$CompiledProtocol,

    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$RawEvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-c4-multimap\evidence',
    [string]$C2Runner = 'tools/run_dtr_carla_c2_rich_scene.ps1',
    [string]$JoinScript = 'research/active/dtr-r0/carla/join_dtr_carla_c4_multimap.py',
    [string]$PlanHelper = 'research/active/dtr-r0/carla/dtr_carla_c4_runner_plan.py',
    [string]$ReuseChildEvidenceRoot = '',
    [ValidateRange(1024, 65533)]
    [int]$BaseRpcPort = 24000,
    [ValidateRange(3, 1024)]
    [int]$PortGroupStride = 3,
    [ValidateRange(120, 7200)]
    [int]$CaptureTimeoutSeconds = 3600,
    [ValidateRange(2.0, 16.0)]
    [double]$MinimumFreePhysicalGB = 4.0,
    [ValidateRange(0, 600)]
    [int]$CooldownSeconds = 20,
    [switch]$Resume,
    [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$script:RunEvidencePath = ''
$script:CompletedGroups = [Collections.Generic.List[string]]::new()
$script:StorageLeaseHelper = Join-Path $PSScriptRoot 'assert_carla_storage_capacity.ps1'
$script:StorageLeaseToken = ''
$script:StorageLeaseCarlaRoot = ''
$script:StorageLeasePythonPath = ''
$script:StorageLeaseOutputRoot = ''

function Assert-StorageLease {
    if ([string]::IsNullOrWhiteSpace($script:StorageLeaseToken)) {
        throw 'CARLA storage lease is unavailable.'
    }
    & $script:StorageLeaseHelper `
        -Action Check `
        -CarlaRoot $script:StorageLeaseCarlaRoot `
        -CarlaPython $script:StorageLeasePythonPath `
        -LeaseToken $script:StorageLeaseToken `
        -OutputRoot $script:StorageLeaseOutputRoot | Out-Null
}

function Release-StorageLease {
    if (-not [string]::IsNullOrWhiteSpace($script:StorageLeaseToken)) {
        & $script:StorageLeaseHelper `
            -Action Release `
            -CarlaRoot $script:StorageLeaseCarlaRoot `
            -CarlaPython $script:StorageLeasePythonPath `
            -LeaseToken $script:StorageLeaseToken | Out-Null
        $script:StorageLeaseToken = ''
    }
}

function Resolve-TaskPath {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$BasePath
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw 'A required path is empty.'
    }
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $Value))
}

function Assert-SafeRunId {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$') {
        throw "RunId is a reserved Windows device name: $Value"
    }
}

function Assert-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is unavailable: $Path"
    }
}

function Get-ContainedRunPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Child
    )
    $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $candidate = [IO.Path]::GetFullPath((Join-Path $rootPath $Child))
    $prefix = $rootPath + [IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Run path escapes its evidence root: $candidate"
    }
    return $candidate
}

function New-ExclusiveDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing evidence overwrite: $Path"
    }
    [IO.Directory]::CreateDirectory($Path) | Out-Null
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    Assert-RequiredFile -Path $Path -Label 'JSON input'
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -Depth 100
}

function Get-JsonText {
    param([Parameter(Mandatory = $true)]$Value)
    return ($Value | ConvertTo-Json -Depth 100) + [Environment]::NewLine
}

function Write-JsonAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing JSON overwrite: $Path"
    }
    $parent = Split-Path -Parent $Path
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporary = "$Path.tmp-$PID-$([Guid]::NewGuid().ToString('N'))"
    try {
        [IO.File]::WriteAllText(
            $temporary,
            (Get-JsonText -Value $Value),
            [Text.UTF8Encoding]::new($false)
        )
        [IO.File]::Move($temporary, $Path)
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

function Assert-SameJsonValue {
    param(
        [Parameter(Mandatory = $true)]$Saved,
        [Parameter(Mandatory = $true)]$Current,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $savedJson = ConvertTo-Json -InputObject $Saved -Depth 100 -Compress
    $currentJson = ConvertTo-Json -InputObject $Current -Depth 100 -Compress
    if ($savedJson -cne $currentJson) {
        throw "Resume plan differs at $Label"
    }
}

function Assert-ExactPath {
    param(
        [Parameter(Mandatory = $true)][string]$Actual,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $actualPath = [IO.Path]::GetFullPath($Actual)
    $expectedPath = [IO.Path]::GetFullPath($Expected)
    if (-not $actualPath.Equals($expectedPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label differs: $actualPath"
    }
}

function Invoke-PlanHelper {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$ProtocolPath
    )
    $output = @(
        & $PythonPath $HelperPath `
            --compiled-protocol $ProtocolPath `
            --base-rpc-port $BaseRpcPort `
            --port-group-stride $PortGroupStride 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
        throw "C4 runner-plan validation failed: $($output -join [Environment]::NewLine)"
    }
    try {
        return ($output -join [Environment]::NewLine) | ConvertFrom-Json -Depth 100
    }
    catch {
        throw "C4 runner-plan helper returned invalid JSON: $($_.Exception.Message)"
    }
}

function Get-PortListeners {
    param([Parameter(Mandatory = $true)][int[]]$Ports)
    @(
        Get-NetTCPConnection `
            -LocalPort $Ports `
            -State Listen `
            -ErrorAction SilentlyContinue
    )
}

function Assert-PortGroupIdle {
    param([Parameter(Mandatory = $true)][int[]]$Ports)
    $listeners = @(Get-PortListeners -Ports $Ports)
    if ($listeners.Count -ne 0) {
        throw "CARLA port group is already in use: $($listeners.LocalPort -join ',')"
    }
}

function Get-StartupEngineIniText {
    param([Parameter(Mandatory = $true)][string]$EngineMapObjectPath)
    $gameMode = '/Game/Carla/Blueprints/Game/CarlaGameMode.CarlaGameMode_C'
    return @(
        '[/Script/EngineSettings.GameMapsSettings]',
        "GameDefaultMap=$EngineMapObjectPath",
        "ServerDefaultMap=$EngineMapObjectPath",
        "TransitionMap=$EngineMapObjectPath",
        "GlobalDefaultGameMode=$gameMode",
        "GlobalDefaultServerGameMode=$gameMode",
        ''
    ) -join "`n"
}

function New-StartupEngineIni {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$EngineMapObjectPath
    )
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing startup EngineIni overwrite: $Path"
    }
    [IO.Directory]::CreateDirectory((Split-Path -Parent $Path)) | Out-Null
    $content = Get-StartupEngineIniText -EngineMapObjectPath $EngineMapObjectPath
    [IO.File]::WriteAllText($Path, $content, [Text.UTF8Encoding]::new($false))
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Assert-StartupEngineIni {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$EngineMapObjectPath,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    Assert-RequiredFile -Path $Path -Label 'C4 startup EngineIni'
    $actualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actualHash -ne $ExpectedSha256) {
        throw "C4 startup EngineIni hash changed: $Path"
    }
    $actual = [IO.File]::ReadAllText($Path, [Text.UTF8Encoding]::new($false))
    $expected = Get-StartupEngineIniText -EngineMapObjectPath $EngineMapObjectPath
    if ($actual -cne $expected) {
        throw "C4 startup EngineIni is not bound to $EngineMapObjectPath"
    }
}

function Copy-FrozenInput {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$SourceBundleRoot,
        [Parameter(Mandatory = $true)][string]$FrozenRoot,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    $relative = [IO.Path]::GetRelativePath($SourceBundleRoot, $SourcePath)
    if (
        [IO.Path]::IsPathRooted($relative) -or
        $relative -eq '..' -or
        $relative.StartsWith(
            '..' + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::Ordinal
        )
    ) {
        throw "C4 input escapes compiled bundle: $SourcePath"
    }
    $destination = Join-Path $FrozenRoot $relative
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    if (Test-Path -LiteralPath $destination) {
        throw "Refusing frozen input overwrite: $destination"
    }
    Copy-Item -LiteralPath $SourcePath -Destination $destination
    $actualHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    if ($actualHash -ne $ExpectedSha256) {
        throw "Frozen input differs from source: $SourcePath"
    }
    return $destination
}

function Initialize-FrozenInputs {
    param([Parameter(Mandatory = $true)]$Plan)
    $frozenRoot = Join-Path $script:RunEvidencePath 'frozen-inputs'
    [IO.Directory]::CreateDirectory($frozenRoot) | Out-Null
    $sourceBundleRoot = [string]$Plan.compiled_bundle_root
    $staticIndex = Copy-FrozenInput `
        -SourcePath ([string]$Plan.compiled_protocol_path) `
        -SourceBundleRoot $sourceBundleRoot `
        -FrozenRoot $frozenRoot `
        -ExpectedSha256 ([string]$Plan.compiled_protocol_sha256)
    $Plan | Add-Member -NotePropertyName frozen_static_index_path -NotePropertyValue $staticIndex

    foreach ($registryName in @('asset_registry', 'scene_registry')) {
        $registry = $Plan.registry_inputs.$registryName
        $frozen = Copy-FrozenInput `
            -SourcePath ([string]$registry.path) `
            -SourceBundleRoot $sourceBundleRoot `
            -FrozenRoot $frozenRoot `
            -ExpectedSha256 ([string]$registry.sha256)
        $registry | Add-Member -NotePropertyName frozen_path -NotePropertyValue $frozen
    }
    $engineRoot = Join-Path $frozenRoot 'startup-engine'
    foreach ($group in @($Plan.map_layout_groups)) {
        $frozenProtocol = Copy-FrozenInput `
            -SourcePath ([string]$group.protocol_path) `
            -SourceBundleRoot $sourceBundleRoot `
            -FrozenRoot $frozenRoot `
            -ExpectedSha256 ([string]$group.protocol_sha256)
        $engineIni = Join-Path $engineRoot "$($group.group_id).Engine.ini"
        $engineHash = New-StartupEngineIni `
            -Path $engineIni `
            -EngineMapObjectPath ([string]$group.engine_ini_map_object_path)
        $group | Add-Member -NotePropertyName runner_protocol_path -NotePropertyValue $frozenProtocol
        $group | Add-Member -NotePropertyName startup_engine_ini_path -NotePropertyValue $engineIni
        $group | Add-Member -NotePropertyName startup_engine_ini_sha256 -NotePropertyValue $engineHash
    }
}

function Assert-ResumePlan {
    param(
        [Parameter(Mandatory = $true)]$SavedPlan,
        [Parameter(Mandatory = $true)]$CurrentPlan
    )
    foreach ($name in @(
        'schema_version',
        'experiment_id',
        'compiled_protocol_sha256',
        'base_rpc_port',
        'port_group_stride',
        'map_count',
        'layout_count',
        'episode_count',
        'group_count',
        'shard_count'
    )) {
        if ([string]$SavedPlan.$name -ne [string]$CurrentPlan.$name) {
            throw "Resume plan differs at $name"
        }
    }
    foreach ($name in @('resolution', 'sensor_order', 'all_ports', 'runtime_admission')) {
        Assert-SameJsonValue `
            -Saved $SavedPlan.$name `
            -Current $CurrentPlan.$name `
            -Label $name
    }
    $frozenRoot = Join-Path $script:RunEvidencePath 'frozen-inputs'
    $staticRelative = [IO.Path]::GetRelativePath(
        [string]$CurrentPlan.compiled_bundle_root,
        [string]$CurrentPlan.compiled_protocol_path
    )
    $expectedStaticPath = Get-ContainedRunPath -Root $frozenRoot -Child $staticRelative
    Assert-ExactPath `
        -Actual ([string]$SavedPlan.frozen_static_index_path) `
        -Expected $expectedStaticPath `
        -Label 'frozen static index path'
    Assert-RequiredFile -Path $expectedStaticPath -Label 'frozen static C4 index'
    if (
        (Get-FileHash -LiteralPath $expectedStaticPath -Algorithm SHA256).Hash -ne
            [string]$SavedPlan.compiled_protocol_sha256
    ) {
        throw 'Frozen static C4 index hash differs.'
    }
    $savedGroups = @($SavedPlan.map_layout_groups)
    $currentGroups = @($CurrentPlan.map_layout_groups)
    if ($savedGroups.Count -ne $currentGroups.Count) {
        throw 'Resume group count differs.'
    }
    for ($index = 0; $index -lt $savedGroups.Count; $index++) {
        foreach ($name in @(
            'group_id',
            'map',
            'startup_map_argument',
            'engine_ini_map_object_path',
            'cold_start_status',
            'relative_protocol_path',
            'protocol_sha256',
            'ordinal',
            'rpc_port'
        )) {
            if ([string]$savedGroups[$index].$name -ne [string]$currentGroups[$index].$name) {
                throw "Resume group $index differs at $name"
            }
        }
        foreach ($name in @('layout_ids', 'episodes', 'ports')) {
            Assert-SameJsonValue `
                -Saved $savedGroups[$index].$name `
                -Current $currentGroups[$index].$name `
                -Label "group $index $name"
        }
        $expectedProtocolPath = Get-ContainedRunPath `
            -Root $frozenRoot `
            -Child ([string]$currentGroups[$index].relative_protocol_path)
        Assert-ExactPath `
            -Actual ([string]$savedGroups[$index].runner_protocol_path) `
            -Expected $expectedProtocolPath `
            -Label "frozen protocol path for $($savedGroups[$index].group_id)"
        Assert-RequiredFile `
            -Path $expectedProtocolPath `
            -Label "frozen protocol for $($savedGroups[$index].group_id)"
        $frozenHash = (
            Get-FileHash `
                -LiteralPath $expectedProtocolPath `
                -Algorithm SHA256
        ).Hash
        if ($frozenHash -ne [string]$savedGroups[$index].protocol_sha256) {
            throw "Frozen protocol hash differs for $($savedGroups[$index].group_id)"
        }
        $expectedEngineIniPath = Get-ContainedRunPath `
            -Root $frozenRoot `
            -Child "startup-engine/$($currentGroups[$index].group_id).Engine.ini"
        Assert-ExactPath `
            -Actual ([string]$savedGroups[$index].startup_engine_ini_path) `
            -Expected $expectedEngineIniPath `
            -Label "startup EngineIni path for $($savedGroups[$index].group_id)"
        Assert-StartupEngineIni `
            -Path $expectedEngineIniPath `
            -EngineMapObjectPath ([string]$savedGroups[$index].engine_ini_map_object_path) `
            -ExpectedSha256 ([string]$savedGroups[$index].startup_engine_ini_sha256)
    }
    foreach ($registryName in @('asset_registry', 'scene_registry')) {
        $savedRegistry = $SavedPlan.registry_inputs.$registryName
        $currentRegistry = $CurrentPlan.registry_inputs.$registryName
        foreach ($name in @('relative_path', 'sha256')) {
            if ([string]$savedRegistry.$name -ne [string]$currentRegistry.$name) {
                throw "Resume $registryName differs at $name"
            }
        }
        $expectedRegistryPath = Get-ContainedRunPath `
            -Root $frozenRoot `
            -Child ([string]$currentRegistry.relative_path)
        Assert-ExactPath `
            -Actual ([string]$savedRegistry.frozen_path) `
            -Expected $expectedRegistryPath `
            -Label "frozen $registryName path"
        Assert-RequiredFile -Path $expectedRegistryPath -Label "frozen $registryName"
        $hash = (Get-FileHash -LiteralPath $expectedRegistryPath -Algorithm SHA256).Hash
        if ($hash -ne [string]$savedRegistry.sha256) {
            throw "Frozen $registryName hash differs."
        }
    }
}

function Get-ChildEvidencePath {
    param([Parameter(Mandatory = $true)]$Group)
    return Join-Path $script:RunEvidencePath "child-evidence/$($Group.group_id)"
}

function Test-CompletedChildAtPath {
    param(
        [Parameter(Mandatory = $true)]$Group,
        [Parameter(Mandatory = $true)][string]$ChildPath
    )
    $resultPath = Join-Path $childPath 'result.json'
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        return $false
    }
    try {
        $result = Read-JsonFile -Path $resultPath
        $checkProperties = @($result.checks.PSObject.Properties)
        $failed = @(
            $checkProperties | Where-Object {
                $_.Value -isnot [bool] -or $_.Value -ne $true
            }
        )
        $frozenProtocol = Join-Path $childPath 'frozen_protocol.json'
        if (
            [string]$result.experiment_id -ne
                'DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2' -or
            [string]$result.status -ne 'DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE' -or
            $checkProperties.Count -eq 0 -or
            $failed.Count -ne 0 -or
            [string]$result.protocol_sha256 -ne [string]$Group.protocol_sha256 -or
            -not (Test-Path -LiteralPath $frozenProtocol -PathType Leaf) -or
            (Get-FileHash -LiteralPath $frozenProtocol -Algorithm SHA256).Hash -ne
                [string]$Group.protocol_sha256
        ) {
            return $false
        }
        return $true
    }
    catch {
        return $false
    }
}

function Test-CompletedChild {
    param([Parameter(Mandatory = $true)]$Group)
    return Test-CompletedChildAtPath `
        -Group $Group `
        -ChildPath (Get-ChildEvidencePath -Group $Group)
}

function Import-ReusableChildren {
    param(
        [Parameter(Mandatory = $true)]$Plan,
        [Parameter(Mandatory = $true)][string]$SourceRunRoot,
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$StorageToolPath
    )
    $sourceRoot = (Resolve-Path -LiteralPath $SourceRunRoot).Path
    $destinationRoot = [IO.Path]::GetFullPath($script:RunEvidencePath)
    if ($sourceRoot -eq $destinationRoot) {
        throw 'Reusable child source must differ from the current C4 evidence root.'
    }
    $sourceChildren = Join-Path $sourceRoot 'child-evidence'
    if (-not (Test-Path -LiteralPath $sourceChildren -PathType Container)) {
        throw "Reusable child evidence directory is unavailable: $sourceChildren"
    }
    $destinationChildren = Join-Path $destinationRoot 'child-evidence'
    [IO.Directory]::CreateDirectory($destinationChildren) | Out-Null
    $receipts = [Collections.Generic.List[object]]::new()
    foreach ($group in @($Plan.map_layout_groups)) {
        $destination = Get-ChildEvidencePath -Group $group
        if (Test-Path -LiteralPath $destination) {
            continue
        }
        $source = Join-Path $sourceChildren ([string]$group.group_id)
        if (
            -not (Test-Path -LiteralPath $source -PathType Container) -or
            -not (Test-CompletedChildAtPath -Group $group -ChildPath $source)
        ) {
            continue
        }
        $sourceResult = Join-Path $source 'result.json'
        $cloneJson = (& $PythonPath $StorageToolPath clone-tree `
            --source $source `
            --destination $destination | Out-String)
        $cloneExitCode = $LASTEXITCODE
        if ($cloneExitCode -ne 0) {
            throw "Reusable child hardlink clone failed with exit code $cloneExitCode"
        }
        $cloneReceipt = $cloneJson | ConvertFrom-Json -Depth 100
        if (-not (Test-CompletedChild -Group $group)) {
            throw "Reusable child copy failed validation: $($group.group_id)"
        }
        $receipts.Add([ordered]@{
            group_id = [string]$group.group_id
            protocol_sha256 = [string]$group.protocol_sha256
            source_child_path = $source
            source_result_sha256 = (
                Get-FileHash -LiteralPath $sourceResult -Algorithm SHA256
            ).Hash
            destination_child_path = $destination
            copy_verified_complete = $true
            materialization = $cloneReceipt
        })
        Write-Output "REUSE verified C4 child $($group.group_id)"
    }
    if ($receipts.Count -ne 0) {
        Write-JsonAtomic `
            -Path (Join-Path $destinationRoot 'reused_children_receipt.json') `
            -Value ([ordered]@{
                schema_version = 'dtr-carla-c4-reused-children-receipt-v1'
                source_run_root = $sourceRoot
                reused_group_count = $receipts.Count
                groups = @($receipts)
            })
    }
}

function Invoke-C2MapGroup {
    param(
        [Parameter(Mandatory = $true)]$Group,
        [Parameter(Mandatory = $true)][string]$C2RunnerPath,
        [Parameter(Mandatory = $true)][string]$CarlaPythonPath,
        [Parameter(Mandatory = $true)][bool]$C2RunnerSupportsResume,
        [Parameter(Mandatory = $true)][string]$StorageLeaseToken
    )
    $ports = @($Group.ports | ForEach-Object { [int]$_ })
    Assert-PortGroupIdle -Ports $ports
    Assert-StartupEngineIni `
        -Path ([string]$Group.startup_engine_ini_path) `
        -EngineMapObjectPath ([string]$Group.engine_ini_map_object_path) `
        -ExpectedSha256 ([string]$Group.startup_engine_ini_sha256)
    $childEvidenceBase = Join-Path $script:RunEvidencePath 'child-evidence'
    [IO.Directory]::CreateDirectory($childEvidenceBase) | Out-Null
    $childPath = Get-ChildEvidencePath -Group $Group
    if (Test-CompletedChild -Group $Group) {
        Write-Output "SKIP complete C4 child $($Group.group_id)"
        $script:CompletedGroups.Add([string]$Group.group_id)
        return
    }
    $partialChild = Test-Path -LiteralPath $childPath
    if ($partialChild -and -not $Resume) {
        throw "Refusing partial child evidence overwrite: $childPath"
    }
    if ($partialChild -and -not $C2RunnerSupportsResume) {
        throw 'The checked-out C2 runner does not support -Resume for partial child evidence.'
    }
    $runnerArguments = @(
        '-NoProfile', '-File', $C2RunnerPath,
        '-RunId', [string]$Group.group_id,
        '-CarlaRoot', $CarlaRoot,
        '-CarlaPython', $CarlaPythonPath,
        '-RawEvidenceRoot', $childEvidenceBase,
        '-Protocol', [string]$Group.runner_protocol_path,
        '-RpcPort', [string]$Group.rpc_port,
        '-StartupEngineIni', [string]$Group.startup_engine_ini_path,
        '-CaptureTimeoutSeconds', [string]$CaptureTimeoutSeconds,
        '-MinimumFreePhysicalGB', [string]$MinimumFreePhysicalGB,
        '-StorageLeaseToken', $StorageLeaseToken
    )
    if ($partialChild) {
        $runnerArguments += '-Resume'
    }
    $primaryFailure = $null
    $cleanupFailure = $null
    try {
        Write-Output (
            "START C4 child group=$($Group.group_id) map=$($Group.map) " +
            "layouts=$($Group.layout_ids -join ',') rpc=$($Group.rpc_port) " +
            'fresh-server-per-sensor resolution=1280x720'
        )
        & pwsh @runnerArguments
        if ($LASTEXITCODE -ne 0) {
            throw "C2 child runner failed for $($Group.group_id) with exit code $LASTEXITCODE"
        }
        if (-not (Test-CompletedChild -Group $Group)) {
            throw "C2 child completion gate failed for $($Group.group_id)"
        }
        $script:CompletedGroups.Add([string]$Group.group_id)
        Write-Output "PASS C4 child $($Group.group_id)"
    }
    catch {
        $primaryFailure = $_
    }
    finally {
        try {
            Assert-PortGroupIdle -Ports $ports
        }
        catch {
            $cleanupFailure = $_
        }
    }
    if ($null -ne $cleanupFailure) {
        if ($null -ne $primaryFailure) {
            throw "$($primaryFailure.Exception.Message) Resource cleanup also failed: $($cleanupFailure.Exception.Message)"
        }
        throw $cleanupFailure
    }
    if ($null -ne $primaryFailure) {
        throw $primaryFailure
    }
    if ($CooldownSeconds -gt 0) {
        Start-Sleep -Seconds $CooldownSeconds
        Assert-PortGroupIdle -Ports $ports
    }
}

function Get-RelativeIndexPath {
    param(
        [Parameter(Mandatory = $true)][string]$IndexDirectory,
        [Parameter(Mandatory = $true)][string]$Target
    )
    return [IO.Path]::GetRelativePath($IndexDirectory, $Target).Replace('\', '/')
}

function New-RuntimeCompiledProtocol {
    param([Parameter(Mandatory = $true)]$Plan)
    $frozenRoot = Join-Path $script:RunEvidencePath 'frozen-inputs'
    $runtimePath = Join-Path $frozenRoot 'runtime-compiled-protocol.json'
    $groups = @()
    foreach ($group in @($Plan.map_layout_groups)) {
        if (-not (Test-CompletedChild -Group $group)) {
            throw "Cannot materialize runtime index before child completion: $($group.group_id)"
        }
        $childPath = Get-ChildEvidencePath -Group $group
        $resultPath = Join-Path $childPath 'result.json'
        $groups += [ordered]@{
            group_id = [string]$group.group_id
            map = [string]$group.map
            startup_map_argument = [string]$group.startup_map_argument
            layout_ids = @($group.layout_ids | ForEach-Object { [string]$_ })
            protocol_path = Get-RelativeIndexPath `
                -IndexDirectory $frozenRoot `
                -Target ([string]$group.runner_protocol_path)
            protocol_sha256 = [string]$group.protocol_sha256
            evidence_path = Get-RelativeIndexPath `
                -IndexDirectory $frozenRoot `
                -Target $childPath
            evidence_result_sha256 = (
                Get-FileHash -LiteralPath $resultPath -Algorithm SHA256
            ).Hash
        }
    }
    $runtime = [ordered]@{
        schema_version = 'dtr-carla-c4-multimap-compiled-v1'
        experiment_id = 'DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1'
        registries = [ordered]@{
            asset_registry = [ordered]@{
                path = Get-RelativeIndexPath `
                    -IndexDirectory $frozenRoot `
                    -Target ([string]$Plan.registry_inputs.asset_registry.frozen_path)
                sha256 = [string]$Plan.registry_inputs.asset_registry.sha256
            }
            scene_registry = [ordered]@{
                path = Get-RelativeIndexPath `
                    -IndexDirectory $frozenRoot `
                    -Target ([string]$Plan.registry_inputs.scene_registry.frozen_path)
                sha256 = [string]$Plan.registry_inputs.scene_registry.sha256
            }
        }
        capture = [ordered]@{
            resolution = @(1280, 720)
            sensor_order = @('instance', 'wearable', 'depth', 'witness')
        }
        admission = [ordered]@{
            expected_map_count = [int]$Plan.runtime_admission.expected_map_count
            expected_protocol_count = [int]$Plan.runtime_admission.expected_protocol_count
            expected_layout_count = [int]$Plan.runtime_admission.expected_layout_count
            expected_episode_count = [int]$Plan.runtime_admission.expected_episode_count
            expected_sensor_count = [int]$Plan.runtime_admission.expected_sensor_count
            expected_shard_count = [int]$Plan.runtime_admission.expected_shard_count
        }
        map_layout_groups = $groups
    }
    $expectedText = Get-JsonText -Value $runtime
    if (Test-Path -LiteralPath $runtimePath -PathType Leaf) {
        $actualText = [IO.File]::ReadAllText($runtimePath, [Text.UTF8Encoding]::new($false))
        if ($actualText -cne $expectedText) {
            throw "Existing runtime compiled protocol differs: $runtimePath"
        }
    }
    else {
        Write-JsonAtomic -Path $runtimePath -Value $runtime
    }
    return $runtimePath
}

function Assert-FinalResult {
    param([Parameter(Mandatory = $true)][string]$FinalRoot)
    $resultPath = Join-Path $FinalRoot 'result.json'
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        return $false
    }
    try {
        $result = Read-JsonFile -Path $resultPath
        $checkProperties = @($result.checks.PSObject.Properties)
        $failed = @(
            $checkProperties | Where-Object {
                $_.Value -isnot [bool] -or $_.Value -ne $true
            }
        )
        $runtimePath = Join-Path `
            $script:RunEvidencePath `
            'frozen-inputs/runtime-compiled-protocol.json'
        if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) {
            return $false
        }
        $runtimeHash = (Get-FileHash -LiteralPath $runtimePath -Algorithm SHA256).Hash
        return (
            [string]$result.experiment_id -eq 'DTR_CARLA_C4_MULTIMAP_WORLD_PACK_V1' -and
            [string]$result.status -eq 'DTR_CARLA_C4_MULTIMAP_SOURCE_COMPLETE' -and
            [string]$result.index_sha256 -eq $runtimeHash -and
            $checkProperties.Count -ne 0 -and
            $failed.Count -eq 0
        )
    }
    catch {
        return $false
    }
}

function Invoke-C4Join {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$JoinPath,
        [Parameter(Mandatory = $true)][string]$RuntimeCompiledProtocol
    )
    $finalRoot = Join-Path $script:RunEvidencePath 'final-package'
    if (Test-Path -LiteralPath $finalRoot) {
        if (Assert-FinalResult -FinalRoot $finalRoot) {
            Write-Host 'SKIP complete C4 final package'
            return $finalRoot
        }
        throw "Refusing partial final-package overwrite: $finalRoot"
    }
    $logRoot = Join-Path $script:RunEvidencePath 'logs'
    [IO.Directory]::CreateDirectory($logRoot) | Out-Null
    $joinStdout = Join-Path $logRoot 'join.stdout.log'
    $joinStderr = Join-Path $logRoot 'join.stderr.log'
    Write-Host 'START C4 final multi-map join and evidence seal'
    & $PythonPath $JoinPath `
        --compiled-protocol $RuntimeCompiledProtocol `
        --output-root $finalRoot `
        1> $joinStdout `
        2> $joinStderr
    if ($LASTEXITCODE -ne 0) {
        throw "C4 final join failed with exit code $LASTEXITCODE"
    }
    if (-not (Assert-FinalResult -FinalRoot $finalRoot)) {
        throw 'C4 final result gate failed.'
    }
    Write-Host 'PASS C4 final multi-map join and evidence seal'
    return $finalRoot
}

try {
    Assert-SafeRunId -Value $RunId
    $carlaPythonPath = Resolve-TaskPath -Value $CarlaPython -BasePath $script:RepoRoot
    $compiledProtocolPath = Resolve-TaskPath -Value $CompiledProtocol -BasePath $script:RepoRoot
    $planHelperPath = Resolve-TaskPath -Value $PlanHelper -BasePath $script:RepoRoot
    foreach ($required in @(
        @($carlaPythonPath, 'CARLA Python'),
        @($compiledProtocolPath, 'static compiled C4 protocol'),
        @($planHelperPath, 'C4 runner-plan helper')
    )) {
        Assert-RequiredFile -Path $required[0] -Label $required[1]
    }
    $currentPlan = Invoke-PlanHelper `
        -PythonPath $carlaPythonPath `
        -HelperPath $planHelperPath `
        -ProtocolPath $compiledProtocolPath
    if ($PlanOnly) {
        $currentPlan | ConvertTo-Json -Depth 100
        exit 0
    }

    $c2RunnerPath = Resolve-TaskPath -Value $C2Runner -BasePath $script:RepoRoot
    $joinScriptPath = Resolve-TaskPath -Value $JoinScript -BasePath $script:RepoRoot
    $storageToolPath = Join-Path `
        $script:RepoRoot `
        'research/active/dtr-r0/carla/carla_storage.py'
    Assert-RequiredFile -Path $c2RunnerPath -Label 'C2 compatibility runner'
    Assert-RequiredFile -Path $joinScriptPath -Label 'C4 final join'
    Assert-RequiredFile -Path $storageToolPath -Label 'CARLA storage tool'
    $runnerCommand = Get-Command -Name $c2RunnerPath -ErrorAction Stop
    foreach ($parameterName in @(
        'RpcPort',
        'StartupEngineIni',
        'MinimumFreePhysicalGB',
        'StorageLeaseToken'
    )) {
        if (-not $runnerCommand.Parameters.ContainsKey($parameterName)) {
            throw "The checked-out C2 runner does not support -$parameterName."
        }
    }
    $c2RunnerSupportsResume = $runnerCommand.Parameters.ContainsKey('Resume')

    foreach ($group in @($currentPlan.map_layout_groups)) {
        Assert-PortGroupIdle -Ports @($group.ports | ForEach-Object { [int]$_ })
    }
    $rawRoot = Resolve-TaskPath -Value $RawEvidenceRoot -BasePath $script:RepoRoot
    $script:RunEvidencePath = Get-ContainedRunPath -Root $rawRoot -Child $RunId
    $script:StorageLeaseCarlaRoot = Resolve-TaskPath `
        -Value $CarlaRoot `
        -BasePath $script:RepoRoot
    $script:StorageLeasePythonPath = $carlaPythonPath
    $script:StorageLeaseOutputRoot = $rawRoot
    $storageLease = & $script:StorageLeaseHelper `
        -Action Acquire `
        -CarlaRoot $script:StorageLeaseCarlaRoot `
        -CarlaPython $carlaPythonPath `
        -ReservationBytes ([long](16GB)) `
        -OutputRoot $rawRoot `
        -LeaseLabel "DTR-CARLA-C4/$RunId" | ConvertFrom-Json -Depth 100
    $script:StorageLeaseToken = [string]$storageLease.lease_token
    [IO.Directory]::CreateDirectory($rawRoot) | Out-Null
    $planPath = Join-Path $script:RunEvidencePath 'runner_plan.json'
    if (Test-Path -LiteralPath $script:RunEvidencePath) {
        if (-not $Resume) {
            throw "Refusing evidence overwrite: $($script:RunEvidencePath)"
        }
        $savedPlan = Read-JsonFile -Path $planPath
        Assert-ResumePlan -SavedPlan $savedPlan -CurrentPlan $currentPlan
        $plan = $savedPlan
    }
    else {
        New-ExclusiveDirectory -Path $script:RunEvidencePath
        $plan = $currentPlan
        Initialize-FrozenInputs -Plan $plan
        Write-JsonAtomic -Path $planPath -Value $plan
        if (-not [string]::IsNullOrWhiteSpace($ReuseChildEvidenceRoot)) {
            $reuseRoot = Resolve-TaskPath `
                -Value $ReuseChildEvidenceRoot `
                -BasePath $script:RepoRoot
            Import-ReusableChildren `
                -Plan $plan `
                -SourceRunRoot $reuseRoot `
                -PythonPath $carlaPythonPath `
                -StorageToolPath $storageToolPath
        }
    }

    foreach ($group in @($plan.map_layout_groups)) {
        Invoke-C2MapGroup `
            -Group $group `
            -C2RunnerPath $c2RunnerPath `
            -CarlaPythonPath $carlaPythonPath `
            -C2RunnerSupportsResume $c2RunnerSupportsResume `
            -StorageLeaseToken $script:StorageLeaseToken
        Assert-StorageLease
    }
    $runtimeCompiledProtocol = New-RuntimeCompiledProtocol -Plan $plan
    $finalRoot = Invoke-C4Join `
        -PythonPath $carlaPythonPath `
        -JoinPath $joinScriptPath `
        -RuntimeCompiledProtocol $runtimeCompiledProtocol
    Assert-StorageLease
    Release-StorageLease
    Write-Output (
        "PASS DTR-CARLA-C4 multi-map source: maps=$($plan.map_count) " +
        "layouts=$($plan.layout_count) sensors=4 shards=$($plan.shard_count) resolution=1280x720"
    )
    Write-Output "evidence: $finalRoot"
    exit 0
}
catch {
    $failure = $_
    if (
        -not [string]::IsNullOrWhiteSpace($script:RunEvidencePath) -and
        (Test-Path -LiteralPath $script:RunEvidencePath -PathType Container)
    ) {
        try {
            $failureReceipt = [ordered]@{
                schema_version = 'dtr-carla-c4-multimap-runner-failure-v1'
                status = 'DTR_CARLA_C4_MULTIMAP_RUNNER_FAILED'
                failed_at_utc = [DateTime]::UtcNow.ToString('o')
                error = $failure.Exception.Message
                completed_groups = @($script:CompletedGroups)
                evidence_preserved = $true
            }
            $failureDirectory = Join-Path $script:RunEvidencePath 'runner-failures'
            $failureName = (
                [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '-' +
                [Guid]::NewGuid().ToString('N') + '.json'
            )
            Write-JsonAtomic `
                -Path (Join-Path $failureDirectory $failureName) `
                -Value $failureReceipt
        }
        catch {}
    }
    try { Release-StorageLease } catch {}
    [Console]::Error.WriteLine("DTR_CARLA_C4_RUNNER_ERROR: $($failure.Exception.Message)")
    exit 2
}
