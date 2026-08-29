[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$RawEvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-c2-rich-scene\evidence',
    [string]$Protocol = 'research/active/dtr-r0/carla/dtr_carla_c2_rich_scene_protocol.json',
    [ValidateRange(120, 7200)]
    [int]$CaptureTimeoutSeconds = 3600
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$script:RpcPort = 2000
$script:CarlaPorts = @(2000, 2001, 2002)
$script:CarlaHost = '127.0.0.1'
$script:StartupTimeoutSeconds = 120
$script:StartupMinimumSeconds = 45
$script:RawRunPath = ''
$script:CarlaInstallRootPath = ''
$script:CarlaPythonPath = ''

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

function Quote-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Contains('"')) {
        throw "Unsupported quote in process argument: $Value"
    }
    return '"' + $Value + '"'
}

function Get-CarlaListeners {
    @(
        Get-NetTCPConnection `
            -LocalPort $script:CarlaPorts `
            -State Listen `
            -ErrorAction SilentlyContinue
    )
}

function Get-CarlaInstallProcesses {
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
    $processes = @(Get-CarlaInstallProcesses)
    if ($processes.Count -ne 0) {
        throw "Packaged CARLA is already running: $($processes.ProcessId -join ',')"
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
        $remaining = @(Get-CarlaInstallProcesses)
        if ($remaining.Count -ne 0) {
            Stop-Process -Id $remaining.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 500
        $listeners = @(Get-CarlaListeners)
    } while (($remaining.Count -ne 0 -or $listeners.Count -ne 0) -and (Get-Date) -lt $deadline)
    $remaining = @(Get-CarlaInstallProcesses)
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
}

function Invoke-SensorCapture {
    param([Parameter(Mandatory = $true)][string]$SensorName)
    Assert-CarlaIdle
    $serverStdout = Join-Path $script:LogRoot "server-$SensorName.stdout.log"
    $serverStderr = Join-Path $script:LogRoot "server-$SensorName.stderr.log"
    $clientStdout = Join-Path $script:LogRoot "client-$SensorName.stdout.log"
    $clientStderr = Join-Path $script:LogRoot "client-$SensorName.stderr.log"
    $serverProcess = $null
    $clientProcess = $null
    $primaryFailure = $null
    $cleanupFailures = [Collections.Generic.List[string]]::new()
    try {
        Write-Output "START C2 1280x720 fresh-server shard $SensorName"
        $serverStartedAt = Get-Date
        $serverProcess = Start-Process `
            -FilePath $script:CarlaExePath `
            -ArgumentList @(
                '-dx12',
                '-RenderOffScreen',
                '-nosound',
                '-quality-level=Epic',
                "-carla-rpc-port=$($script:RpcPort)"
            ) `
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
    $resolution = @($protocolValue.capture.resolution | ForEach-Object { [int]$_ })
    if (($resolution -join 'x') -ne '1280x720') {
        throw "C2 formal resolution must be 1280x720: $($resolution -join 'x')"
    }
    $sensorOrder = @($protocolValue.capture.sensor_order | ForEach-Object { [string]$_ })
    if (($sensorOrder -join ',') -ne 'instance,wearable,depth,witness') {
        throw "Unexpected C2 sensor order: $($sensorOrder -join ',')"
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
    Assert-CarlaIdle
    [IO.Directory]::CreateDirectory($rawRoot) | Out-Null
    New-ExclusiveDirectory -Path $script:RawRunPath
    $script:LogRoot = Join-Path $script:RawRunPath 'logs'
    [IO.Directory]::CreateDirectory($script:LogRoot) | Out-Null

    foreach ($sensorName in $sensorOrder) {
        Invoke-SensorCapture -SensorName $sensorName
    }
    Assert-CarlaIdle
    Invoke-Join
    Write-Output (
        "PASS DTR-CARLA-C2 rich source: sensors=4 resolution=1280x720 " +
        "frames_per_sensor=$($script:ExpectedFramesPerSensor) unique_blueprints>=$($script:MinimumUniqueBlueprints)"
    )
    Write-Output "evidence: $($script:RawRunPath)"
    exit 0
}
catch {
    try { Stop-OwnedPython -Process $null } catch {}
    try { Stop-OwnedCarla } catch {}
    [Console]::Error.WriteLine("DTR_CARLA_C2_RUNNER_ERROR: $($_.Exception.Message)")
    exit 2
}
