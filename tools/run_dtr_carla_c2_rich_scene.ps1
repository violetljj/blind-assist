[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$RawEvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-c2-rich-scene\evidence',
    [string]$Protocol = 'research/active/dtr-r0/carla/dtr_carla_c2_rich_scene_protocol.json',
    [string]$StartupEngineIni = '',
    [ValidateRange(1024, 65533)]
    [int]$RpcPort = 2000,
    [ValidateRange(120, 7200)]
    [int]$CaptureTimeoutSeconds = 3600,
    [ValidateRange(2.0, 16.0)]
    [double]$MinimumFreePhysicalGB = 4.0,
    [ValidateRange(1073741824, 8589934592)]
    [long]$StorageReservationBytes = 8589934592,
    [string]$StorageLeaseToken = '',
    [switch]$VisualShellSourceProbeOnly,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$script:RpcPort = $RpcPort
$script:CarlaPorts = @($RpcPort, $RpcPort + 1, $RpcPort + 2)
$script:CarlaHost = '127.0.0.1'
$script:StartupTimeoutSeconds = 120
$script:StartupMinimumSeconds = 45
$script:RenderQualityLevel = 'Epic'
$script:RenderBackend = 'dx12'
$script:MinimumFreePhysicalGB = $MinimumFreePhysicalGB
$script:CapacityTimeoutSeconds = 300
$script:RawRunPath = ''
$script:CarlaInstallRootPath = ''
$script:CarlaPythonPath = ''
$script:StartupEngineIniPath = ''
$script:StorageLeaseHelper = Join-Path $PSScriptRoot 'assert_carla_storage_capacity.ps1'
$script:StorageLeaseToken = $StorageLeaseToken
$script:StorageLeaseCarlaRoot = ''
$script:StorageLeaseOutputRoot = ''
$script:OwnsStorageLease = $false

function Assert-StorageLease {
    if ([string]::IsNullOrWhiteSpace($script:StorageLeaseToken)) {
        throw 'CARLA storage lease is unavailable.'
    }
    & $script:StorageLeaseHelper `
        -Action Check `
        -CarlaRoot $script:StorageLeaseCarlaRoot `
        -CarlaPython $script:CarlaPythonPath `
        -LeaseToken $script:StorageLeaseToken `
        -OutputRoot $script:StorageLeaseOutputRoot | Out-Null
}

function Release-StorageLease {
    if ($script:OwnsStorageLease -and -not [string]::IsNullOrWhiteSpace($script:StorageLeaseToken)) {
        & $script:StorageLeaseHelper `
            -Action Release `
            -CarlaRoot $script:StorageLeaseCarlaRoot `
            -CarlaPython $script:CarlaPythonPath `
            -LeaseToken $script:StorageLeaseToken | Out-Null
        $script:StorageLeaseToken = ''
        $script:OwnsStorageLease = $false
    }
}

function Resolve-LocalPath {
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
        throw "Refusing partial evidence or overwrite: $Path"
    }
    [IO.Directory]::CreateDirectory($Path) | Out-Null
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    Assert-RequiredFile -Path $Path -Label 'JSON input'
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -Depth 100
}

function Clear-NonTerminalJoinArtifacts {
    param([Parameter(Mandatory = $true)][string]$RunRoot)
    $resultPath = Get-ContainedRunPath -Root $RunRoot -Child 'result.json'
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        return
    }
    $result = Read-JsonFile -Path $resultPath
    if ([string]$result.status -eq 'DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE') {
        throw "Refusing to resume a terminal joined run: $RunRoot"
    }
    foreach ($relative in @(
        'model',
        'evaluator',
        'sealed_model_manifest.json',
        'sealed_evidence_manifest.json',
        'result.json'
    )) {
        $target = Get-ContainedRunPath -Root $RunRoot -Child $relative
        if (Test-Path -LiteralPath $target -PathType Container) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
        elseif (Test-Path -LiteralPath $target -PathType Leaf) {
            Remove-Item -LiteralPath $target -Force
        }
    }
    Write-Output 'CLEAR nonterminal C2 join artifacts; immutable sensor shards preserved'
}

function Quote-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Contains('"')) {
        throw "Unsupported quote in process argument: $Value"
    }
    return '"' + $Value + '"'
}

function Get-ExpectedEngineMapObjectPath {
    param([Parameter(Mandatory = $true)][string]$MapName)
    $normalized = $MapName.TrimStart('/')
    if ($normalized -notmatch '^Carla/Maps/(?<leaf>[A-Za-z0-9_]+)$') {
        throw "Unsupported CARLA startup map identity: $MapName"
    }
    return "/Game/$normalized.$($Matches.leaf)"
}

function Assert-StartupEngineIniMap {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$MapName
    )
    Assert-RequiredFile -Path $Path -Label 'CARLA startup Engine.ini'
    $expectedObjectPath = Get-ExpectedEngineMapObjectPath -MapName $MapName
    $content = Get-Content -LiteralPath $Path -Raw
    foreach ($key in @('GameDefaultMap', 'ServerDefaultMap', 'TransitionMap')) {
        $matches = @(
            [regex]::Matches(
                $content,
                "(?m)^$([regex]::Escape($key))=(?<value>[^\r\n]+)$"
            )
        )
        if ($matches.Count -ne 1) {
            throw "Startup Engine.ini must define $key exactly once: $Path"
        }
        if ([string]$matches[0].Groups['value'].Value -ne $expectedObjectPath) {
            throw (
                "Startup Engine.ini $key does not bind protocol map: " +
                "$($matches[0].Groups['value'].Value) != $expectedObjectPath"
            )
        }
    }
}

function New-RuntimeStartupEngineIni {
    param([Parameter(Mandatory = $true)][string]$SensorName)
    Assert-StartupEngineIniMap `
        -Path $script:StartupEngineIniPath `
        -MapName ([string]$script:ProtocolMap)
    $leaf = (
        "blindassist-c2-$PID-$($script:RpcPort)-$SensorName-" +
        "$([Guid]::NewGuid().ToString('N')).Engine.ini"
    )
    $path = Join-Path ([IO.Path]::GetTempPath()) $leaf
    Copy-Item -LiteralPath $script:StartupEngineIniPath -Destination $path
    Assert-StartupEngineIniMap `
        -Path $path `
        -MapName ([string]$script:ProtocolMap)
    return $path
}

function Remove-RuntimeStartupEngineIni {
    param([string]$Path)
    if (-not [string]::IsNullOrWhiteSpace($Path) -and (Test-Path -LiteralPath $Path)) {
        Remove-Item -LiteralPath $Path -Force
    }
}

function Get-CarlaListeners {
    @(
        Get-NetTCPConnection `
            -LocalPort $script:CarlaPorts `
            -State Listen `
            -ErrorAction SilentlyContinue
    )
}

function Get-OwnedCarlaProcesses {
    $prefix = $script:CarlaInstallRootPath.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $portArgument = "-carla-rpc-port=$($script:RpcPort)"
    @(
        Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                $executable = [string]$_.ExecutablePath
                $commandLine = [string]$_.CommandLine
                -not [string]::IsNullOrWhiteSpace($executable) -and
                $executable.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -and
                $commandLine.IndexOf($portArgument, [StringComparison]::OrdinalIgnoreCase) -ge 0
            }
    )
}

function Get-AllCarlaProcesses {
    $prefix = $script:CarlaInstallRootPath.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    @(
        Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                $executable = [string]$_.ExecutablePath
                -not [string]::IsNullOrWhiteSpace($executable) -and
                $executable.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
            }
    )
}

function Wait-CarlaCapacity {
    $deadline = (Get-Date).AddSeconds($script:CapacityTimeoutSeconds)
    $stableSeconds = 0
    do {
        $processes = @(Get-AllCarlaProcesses)
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        $freePhysicalGB = [double]$os.FreePhysicalMemory / 1MB
        if (
            $processes.Count -eq 0 -and
            $freePhysicalGB -ge $script:MinimumFreePhysicalGB
        ) {
            $stableSeconds += 1
            if ($stableSeconds -ge 5) {
                return
            }
        }
        else {
            $stableSeconds = 0
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    throw (
        'CARLA capacity did not recover: processes=' +
        $processes.Count + '; free_physical_gb=' +
        ([Math]::Round($freePhysicalGB, 2))
    )
}

function Get-OwnedPythonProcesses {
    if ([string]::IsNullOrWhiteSpace($script:RawRunPath)) {
        return @()
    }
    @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $executable = [string]$_.ExecutablePath
                $commandLine = [string]$_.CommandLine
                $executable.Equals(
                    $script:CarlaPythonPath,
                    [StringComparison]::OrdinalIgnoreCase
                ) -and
                $commandLine.IndexOf(
                    $script:RawRunPath,
                    [StringComparison]::OrdinalIgnoreCase
                ) -ge 0
            }
    )
}

function Assert-CarlaIdle {
    $listeners = @(Get-CarlaListeners)
    if ($listeners.Count -ne 0) {
        throw "CARLA ports are already in use: $($listeners.LocalPort -join ',')"
    }
    $processes = @(Get-OwnedCarlaProcesses)
    if ($processes.Count -ne 0) {
        throw "CARLA is already running on RPC port $($script:RpcPort): $($processes.ProcessId -join ',')"
    }
    $allProcesses = @(Get-AllCarlaProcesses)
    if ($allProcesses.Count -ne 0) {
        throw "Another CARLA allocation is active: $($allProcesses.ProcessId -join ',')"
    }
}

function Stop-OwnedPython {
    param([AllowNull()][System.Diagnostics.Process]$Process)
    $ids = @()
    if ($null -ne $Process -and -not $Process.HasExited) {
        $ids += $Process.Id
    }
    $ids += @(Get-OwnedPythonProcesses | ForEach-Object { [int]$_.ProcessId })
    $ids = @($ids | Sort-Object -Unique)
    if ($ids.Count -ne 0) {
        Stop-Process -Id $ids -Force -ErrorAction SilentlyContinue
    }
    $deadline = (Get-Date).AddSeconds(15)
    do {
        $remaining = @(Get-OwnedPythonProcesses)
        if ($remaining.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw 'Task-owned C2 Python process remained after cleanup.'
}

function Stop-OwnedCarla {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        $remaining = @(Get-OwnedCarlaProcesses)
        if ($remaining.Count -ne 0) {
            Stop-Process -Id $remaining.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 500
        $listeners = @(Get-CarlaListeners)
    } while (($remaining.Count -ne 0 -or $listeners.Count -ne 0) -and (Get-Date) -lt $deadline)
    $remaining = @(Get-OwnedCarlaProcesses)
    $listeners = @(Get-CarlaListeners)
    if ($remaining.Count -ne 0 -or $listeners.Count -ne 0) {
        throw (
            'CARLA cleanup verification failed: processes=' +
            ($remaining.ProcessId -join ',') + '; listeners=' +
            ($listeners.LocalPort -join ',')
        )
    }
}

function Wait-CarlaReady {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$ServerProcess)
    $deadline = (Get-Date).AddSeconds($script:StartupTimeoutSeconds)
    do {
        $rpc = @(Get-CarlaListeners | Where-Object { [int]$_.LocalPort -eq $script:RpcPort })
        if ($rpc.Count -ne 0) {
            return
        }
        $ServerProcess.Refresh()
        if ($ServerProcess.HasExited) {
            throw 'CARLA launcher exited before the RPC port opened.'
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    throw 'CARLA startup timed out.'
}

function Assert-ShardResult {
    param(
        [Parameter(Mandatory = $true)][string]$SensorName,
        [Parameter(Mandatory = $true)][int]$ExitCode
    )
    $path = Join-Path $script:RawRunPath "shards/$SensorName/result.json"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "$SensorName capture exited without result.json (exit=$ExitCode)"
    }
    $result = Read-JsonFile -Path $path
    $failed = @($result.checks.PSObject.Properties | Where-Object { -not [bool]$_.Value })
    if (
        $ExitCode -ne 0 -or
        [string]$result.status -ne 'DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE' -or
        $failed.Count -ne 0 -or
        [int]$result.payload_count -ne [int]$script:ExpectedFramesPerSensor -or
        [int]$result.unique_actual_blueprint_count -lt [int]$script:MinimumUniqueBlueprints
    ) {
        throw (
            "$SensorName shard gate failed: exit=$ExitCode status=$($result.status) " +
            "failed_checks=$($failed.Name -join ',')"
        )
    }
    $protocolHash = (Get-FileHash -LiteralPath $script:ProtocolPath -Algorithm SHA256).Hash
    $captureHash = (Get-FileHash -LiteralPath $script:CaptureScriptPath -Algorithm SHA256).Hash
    if (
        [string]$result.protocol_sha256 -ne $protocolHash -or
        [string]$result.capture_script_sha256 -ne $captureHash
    ) {
        throw "$SensorName shard source hash mismatch during resume validation."
    }
}

function Invoke-SensorCapture {
    param([Parameter(Mandatory = $true)][string]$SensorName)
    Wait-CarlaCapacity
    Assert-CarlaIdle
    $serverStdout = Join-Path $script:LogRoot "server-$SensorName.stdout.log"
    $serverStderr = Join-Path $script:LogRoot "server-$SensorName.stderr.log"
    $clientStdout = Join-Path $script:LogRoot "client-$SensorName.stdout.log"
    $clientStderr = Join-Path $script:LogRoot "client-$SensorName.stderr.log"
    $serverProcess = $null
    $clientProcess = $null
    $runtimeStartupEngineIniPath = ''
    $primaryFailure = $null
    $cleanupFailures = [Collections.Generic.List[string]]::new()
    try {
        Write-Output "START C2 1280x720 fresh-server shard $SensorName"
        $serverStartedAt = Get-Date
        $serverArguments = [Collections.Generic.List[string]]::new()
        if (-not [string]::IsNullOrWhiteSpace($script:StartupEngineIniPath)) {
            $runtimeStartupEngineIniPath = New-RuntimeStartupEngineIni `
                -SensorName $SensorName
            $serverArguments.Add("-EngineIni=$runtimeStartupEngineIniPath")
        }
        $serverArguments.Add("-$($script:RenderBackend)")
        $serverArguments.Add('-RenderOffScreen')
        $serverArguments.Add('-nosound')
        $serverArguments.Add("-quality-level=$($script:RenderQualityLevel)")
        $serverArguments.Add("-carla-rpc-port=$($script:RpcPort)")
        $serverProcess = Start-Process `
            -FilePath $script:CarlaExePath `
            -ArgumentList $serverArguments.ToArray() `
            -WorkingDirectory $script:CarlaInstallRootPath `
            -WindowStyle Hidden `
            -RedirectStandardOutput $serverStdout `
            -RedirectStandardError $serverStderr `
            -PassThru
        Wait-CarlaReady -ServerProcess $serverProcess
        $remainingStartup = (
            $script:StartupMinimumSeconds -
            [int]((Get-Date) - $serverStartedAt).TotalSeconds
        )
        if ($remainingStartup -gt 0) {
            Start-Sleep -Seconds $remainingStartup
        }
        $clientProcess = Start-Process `
            -FilePath $script:CarlaPythonPath `
            -ArgumentList @(
                (Quote-ProcessArgument $script:CaptureScriptPath),
                '--protocol', (Quote-ProcessArgument $script:ProtocolPath),
                '--output-root', (Quote-ProcessArgument $script:RawRunPath),
                '--sensor', $SensorName,
                '--host', $script:CarlaHost,
                '--port', [string]$script:RpcPort
            ) `
            -WorkingDirectory $script:RepoRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $clientStdout `
            -RedirectStandardError $clientStderr `
            -PassThru
        $deadline = (Get-Date).AddSeconds($CaptureTimeoutSeconds)
        while (-not $clientProcess.WaitForExit(1000) -and (Get-Date) -lt $deadline) {
            $rpc = @(Get-CarlaListeners | Where-Object { [int]$_.LocalPort -eq $script:RpcPort })
            if ($rpc.Count -eq 0) {
                throw "CARLA exited during $SensorName capture."
            }
        }
        if (-not $clientProcess.HasExited) {
            throw "$SensorName capture exceeded $CaptureTimeoutSeconds seconds."
        }
        $clientProcess.WaitForExit()
        $clientProcess.Refresh()
        Assert-ShardResult -SensorName $SensorName -ExitCode $clientProcess.ExitCode
        Write-Output "PASS C2 shard $SensorName"
    }
    catch {
        $primaryFailure = $_
    }
    finally {
        try { Stop-OwnedPython -Process $clientProcess } catch { $cleanupFailures.Add($_.Exception.Message) }
        try { Stop-OwnedCarla } catch { $cleanupFailures.Add($_.Exception.Message) }
        try {
            Remove-RuntimeStartupEngineIni -Path $runtimeStartupEngineIniPath
        }
        catch {
            $cleanupFailures.Add($_.Exception.Message)
        }
    }
    if ($cleanupFailures.Count -ne 0) {
        $cleanupText = $cleanupFailures -join '; '
        if ($null -ne $primaryFailure) {
            throw "$($primaryFailure.Exception.Message) Cleanup also failed: $cleanupText"
        }
        throw $cleanupText
    }
    if ($null -ne $primaryFailure) {
        throw $primaryFailure
    }
    Wait-CarlaCapacity
    Assert-CarlaIdle
}

function Invoke-Join {
    $stdout = Join-Path $script:LogRoot 'join.stdout.log'
    $stderr = Join-Path $script:LogRoot 'join.stderr.log'
    Write-Output 'START C2 model/evaluator join and evidence seal'
    $process = Start-Process `
        -FilePath $script:CarlaPythonPath `
        -ArgumentList @(
            (Quote-ProcessArgument $script:JoinScriptPath),
            '--protocol', (Quote-ProcessArgument $script:ProtocolPath),
            '--root', (Quote-ProcessArgument $script:RawRunPath)
        ) `
        -WorkingDirectory $script:RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    if (-not $process.WaitForExit(2400000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw 'C2 join exceeded 2400 seconds.'
    }
    $process.WaitForExit()
    $process.Refresh()
    $path = Join-Path $script:RawRunPath 'result.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "C2 join exited without result.json (exit=$($process.ExitCode))"
    }
    $result = Read-JsonFile -Path $path
    $failed = @($result.checks.PSObject.Properties | Where-Object { -not [bool]$_.Value })
    if (
        $process.ExitCode -ne 0 -or
        [string]$result.status -ne 'DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE' -or
        $failed.Count -ne 0
    ) {
        throw (
            "C2 joined gate failed: exit=$($process.ExitCode) status=$($result.status) " +
            "failed_checks=$($failed.Name -join ',')"
        )
    }
    Write-Output 'PASS C2 model/evaluator join and evidence seal'
}

try {
    Assert-SafeRunId -Value $RunId
    $script:CarlaLibraryRootPath = Resolve-LocalPath -Value $CarlaRoot -BasePath $script:RepoRoot
    $script:CarlaInstallRootPath = Join-Path $script:CarlaLibraryRootPath '0.9.16'
    $script:CarlaInstallRootPath = (Resolve-Path -LiteralPath $script:CarlaInstallRootPath).Path
    $script:CarlaExePath = Join-Path $script:CarlaInstallRootPath 'CarlaUE4.exe'
    $script:CarlaPythonPath = Resolve-LocalPath -Value $CarlaPython -BasePath $script:RepoRoot
    $script:ProtocolPath = Resolve-LocalPath -Value $Protocol -BasePath $script:RepoRoot
    if (-not [string]::IsNullOrWhiteSpace($StartupEngineIni)) {
        $script:StartupEngineIniPath = Resolve-LocalPath `
            -Value $StartupEngineIni `
            -BasePath $script:RepoRoot
    }
    $script:CaptureScriptPath = Join-Path $script:RepoRoot 'research/active/dtr-r0/carla/capture_dtr_carla_c2_rich_scene.py'
    $script:JoinScriptPath = Join-Path $script:RepoRoot 'research/active/dtr-r0/carla/join_dtr_carla_c2_rich_scene.py'
    foreach ($required in @(
        @($script:CarlaExePath, 'CARLA launcher'),
        @($script:CarlaPythonPath, 'CARLA Python'),
        @($script:ProtocolPath, 'C2 protocol'),
        @($script:CaptureScriptPath, 'C2 capture'),
        @($script:JoinScriptPath, 'C2 join')
    )) {
        Assert-RequiredFile -Path $required[0] -Label $required[1]
    }
    $protocolValue = Read-JsonFile -Path $script:ProtocolPath
    if ([string]$protocolValue.experiment_id -ne 'DTR_CARLA_C2_RICH_MULTILAYOUT_OCCLUSION_SOURCE_V2') {
        throw "Unexpected C2 protocol identity: $($protocolValue.experiment_id)"
    }
    $script:ProtocolMap = [string]$protocolValue.environment.map
    if (-not [string]::IsNullOrWhiteSpace($script:StartupEngineIniPath)) {
        Assert-StartupEngineIniMap `
            -Path $script:StartupEngineIniPath `
            -MapName $script:ProtocolMap
    }
    $resolution = @($protocolValue.capture.resolution | ForEach-Object { [int]$_ })
    if (($resolution -join 'x') -ne '1280x720') {
        throw "C2 formal resolution must be 1280x720: $($resolution -join 'x')"
    }
    $sensorOrder = @($protocolValue.capture.sensor_order | ForEach-Object { [string]$_ })
    if (($sensorOrder -join ',') -ne 'instance,wearable,depth,witness') {
        throw "Unexpected C2 sensor order: $($sensorOrder -join ',')"
    }
    if ($VisualShellSourceProbeOnly) {
        if ($Resume) { throw 'The bounded visual-shell probe cannot resume a consumed capture.' }
        if ($protocolValue.final_visual_shell_probe.schema -ne 'blindassist-dtr-final-visual-shell-probe-v1' -or
            $protocolValue.final_visual_shell_probe.method_predictions_or_scores_allowed -ne $false -or
            $protocolValue.final_visual_shell_probe.probe_pixels_reusable_as_fit_or_final -ne $false) {
            throw 'The visual-shell switch requires the isolated source-only protocol.'
        }
        $sensorOrder = @('instance', 'witness')
    }
    if ($null -ne $protocolValue.capture.render_quality_level) {
        $script:RenderQualityLevel = [string]$protocolValue.capture.render_quality_level
    }
    if ($script:RenderQualityLevel -notin @('Low', 'Epic')) {
        throw "Unsupported CARLA render quality: $($script:RenderQualityLevel)"
    }
    if ($null -ne $protocolValue.capture.render_backend) {
        $script:RenderBackend = [string]$protocolValue.capture.render_backend
    }
    if ($script:RenderBackend -notin @('dx11', 'dx12')) {
        throw "Unsupported CARLA render backend: $($script:RenderBackend)"
    }
    $script:ExpectedFramesPerSensor = 0
    foreach ($scenario in @($protocolValue.scenarios)) {
        $layout = $protocolValue.layouts.PSObject.Properties[$scenario.layout_id].Value
        $script:ExpectedFramesPerSensor += [int][Math]::Round(
            [double]$layout.duration_seconds / [double]$protocolValue.environment.sample_seconds
        ) + 1
    }
    $script:MinimumUniqueBlueprints = [int]$protocolValue.admission.minimum_unique_actual_blueprints_across_pack
    $rawRoot = Resolve-LocalPath -Value $RawEvidenceRoot -BasePath $script:RepoRoot
    $script:RawRunPath = Get-ContainedRunPath -Root $rawRoot -Child $RunId
    Wait-CarlaCapacity
    Assert-CarlaIdle
    $script:StorageLeaseCarlaRoot = $script:CarlaLibraryRootPath
    $script:StorageLeaseOutputRoot = $rawRoot
    if ([string]::IsNullOrWhiteSpace($script:StorageLeaseToken)) {
        $storageLease = & $script:StorageLeaseHelper `
            -Action Acquire `
            -CarlaRoot $script:CarlaLibraryRootPath `
            -CarlaPython $script:CarlaPythonPath `
            -ReservationBytes $StorageReservationBytes `
            -OutputRoot $rawRoot `
            -LeaseLabel "DTR-CARLA-C2/$RunId" | ConvertFrom-Json -Depth 100
        $script:StorageLeaseToken = [string]$storageLease.lease_token
        $script:OwnsStorageLease = $true
    }
    else {
        Assert-StorageLease
    }
    [IO.Directory]::CreateDirectory($rawRoot) | Out-Null
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $script:RawRunPath -PathType Container)) {
            throw "Resume run is unavailable: $($script:RawRunPath)"
        }
        Clear-NonTerminalJoinArtifacts -RunRoot $script:RawRunPath
    }
    else {
        New-ExclusiveDirectory -Path $script:RawRunPath
    }
    $script:LogRoot = Join-Path $script:RawRunPath 'logs'
    [IO.Directory]::CreateDirectory($script:LogRoot) | Out-Null

    foreach ($sensorName in $sensorOrder) {
        $shardRoot = Join-Path $script:RawRunPath "shards/$sensorName"
        $shardResult = Join-Path $shardRoot 'result.json'
        if ($Resume -and (Test-Path -LiteralPath $shardResult -PathType Leaf)) {
            Assert-ShardResult -SensorName $sensorName -ExitCode 0
            Write-Output "SKIP verified completed C2 shard $sensorName"
            Assert-StorageLease
            continue
        }
        if ($Resume -and (Test-Path -LiteralPath $shardRoot -PathType Container)) {
            $partial = @(Get-ChildItem -LiteralPath $shardRoot -Force)
            if ($partial.Count -ne 0) {
                throw "Refusing non-empty partial shard during resume: $shardRoot"
            }
            Remove-Item -LiteralPath $shardRoot
        }
        Invoke-SensorCapture -SensorName $sensorName
        Assert-StorageLease
        if ($VisualShellSourceProbeOnly) {
            $gateArgs = @(
                (Join-Path $script:RepoRoot 'research/active/dtr-r0/carla/evaluate_dtr_final_visual_shell_probe.py'),
                '--protocol', $script:ProtocolPath, '--root', $script:RawRunPath,
                '--output', (Join-Path $script:RawRunPath "source-gate-$sensorName.json")
            )
            if ($sensorName -eq 'witness') { $gateArgs += '--require-witness' }
            & $script:CarlaPythonPath @gateArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Visual-shell source gate failed after $sensorName; no next shard or final capture is authorized."
            }
        }
    }
    Assert-CarlaIdle
    Assert-StorageLease
    if ($VisualShellSourceProbeOnly) {
        Release-StorageLease
        Write-Output "PASS bounded visual-shell source probe; remaining full-roster gates still required: $($script:RawRunPath)"
        exit 0
    }
    Invoke-Join
    Assert-StorageLease
    Release-StorageLease
    Write-Output (
        "PASS DTR-CARLA-C2 rich source: sensors=4 resolution=1280x720 " +
        "frames_per_sensor=$($script:ExpectedFramesPerSensor) unique_blueprints>=$($script:MinimumUniqueBlueprints)"
    )
    Write-Output "evidence: $($script:RawRunPath)"
    exit 0
}
catch {
    $failure = $_
    try { Stop-OwnedPython -Process $null } catch {}
    try { Stop-OwnedCarla } catch {}
    try { Release-StorageLease } catch {}
    [Console]::Error.WriteLine("DTR_CARLA_C2_RUNNER_ERROR: $($failure.Exception.Message)")
    exit 2
}
