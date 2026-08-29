[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$RawEvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-c1-complex\evidence',
    [string]$Protocol = 'research/active/dtr-r0/carla/dtr_carla_c1_complex_protocol.json',
    [ValidateRange(60, 7200)]
    [int]$CaptureTimeoutSeconds = 2400
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$script:RpcPort = 2000
$script:CarlaPorts = @(2000, 2001, 2002)
$script:StartupTimeoutSeconds = 120
$script:StartupMinimumSeconds = 45
$script:CarlaHost = '127.0.0.1'

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
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing partial evidence or overwrite; $Label already exists: $Path"
    }
    [IO.Directory]::CreateDirectory($Path) | Out-Null
}

function Read-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    Assert-RequiredFile -Path $Path -Label $Label
    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -Depth 100
    }
    catch {
        throw "$Label is not valid JSON: $Path ($($_.Exception.Message))"
    }
}

function Quote-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Contains('"')) {
        throw "A process argument contains an unsupported quote: $Value"
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

function Get-OwnedCaptureProcesses {
    @(
        Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object {
                $executable = [string]$_.ExecutablePath
                $commandLine = [string]$_.CommandLine
                $executable.Equals(
                    $script:CarlaPythonPath,
                    [StringComparison]::OrdinalIgnoreCase
                ) -and
                $commandLine.IndexOf(
                    $script:CaptureScriptPath,
                    [StringComparison]::OrdinalIgnoreCase
                ) -ge 0 -and
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
        $details = @(
            $listeners |
                Sort-Object LocalPort, OwningProcess |
                ForEach-Object { "$($_.LocalPort)/pid=$($_.OwningProcess)" }
        ) -join ', '
        throw "Refusing to share CARLA ports 2000-2002; listeners: $details"
    }
    $processes = @(Get-CarlaInstallProcesses)
    if ($processes.Count -ne 0) {
        throw "Refusing to share packaged CARLA; process ids: $($processes.ProcessId -join ', ')"
    }
}

function Stop-OwnedCapture {
    param([AllowNull()][System.Diagnostics.Process]$ClientProcess)

    $ownedIds = @()
    if ($null -ne $ClientProcess -and -not $ClientProcess.HasExited) {
        $ownedIds += $ClientProcess.Id
    }
    $ownedIds += @(Get-OwnedCaptureProcesses | ForEach-Object { [int]$_.ProcessId })
    $ownedIds = @($ownedIds | Sort-Object -Unique)
    if ($ownedIds.Count -ne 0) {
        Stop-Process -Id $ownedIds -Force -ErrorAction SilentlyContinue
    }
    $deadline = (Get-Date).AddSeconds(10)
    do {
        $remaining = @(Get-OwnedCaptureProcesses)
        $clientStillRunning = $null -ne $ClientProcess -and -not $ClientProcess.HasExited
        if (-not $clientStillRunning -and $remaining.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw 'Task-owned C1 capture process remained after cleanup.'
}

function Stop-OwnedCarla {
    $deadline = (Get-Date).AddSeconds(20)
    do {
        $remaining = @(Get-CarlaInstallProcesses)
        if ($remaining.Count -ne 0) {
            Stop-Process -Id $remaining.ProcessId -Force -ErrorAction SilentlyContinue
        }
        $listeners = @(Get-CarlaListeners)
        if ($remaining.Count -eq 0 -and $listeners.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    $remaining = @(Get-CarlaInstallProcesses)
    $listeners = @(Get-CarlaListeners)
    throw (
        'CARLA cleanup verification failed: processes=' +
        ($remaining.ProcessId -join ',') + '; listeners=' +
        ($listeners.LocalPort -join ',')
    )
}

function Wait-CarlaReady {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$ServerProcess)
    $deadline = (Get-Date).AddSeconds($script:StartupTimeoutSeconds)
    do {
        $rpcListeners = @(
            Get-CarlaListeners |
                Where-Object { [int]$_.LocalPort -eq $script:RpcPort }
        )
        if ($rpcListeners.Count -ne 0) {
            return
        }
        $ServerProcess.Refresh()
        if ($ServerProcess.HasExited) {
            throw "CARLA launcher exited before opening port $($script:RpcPort)"
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    throw "CARLA did not open port $($script:RpcPort) within $($script:StartupTimeoutSeconds) seconds"
}

function Assert-CaptureResult {
    param(
        [Parameter(Mandatory = $true)][string]$SensorName,
        [Parameter(Mandatory = $true)][int]$ClientExitCode
    )
    $resultPath = Join-Path $script:RawRunPath "shards/$SensorName/result.json"
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "$SensorName shard exited with code $ClientExitCode without result.json"
    }
    $result = Read-JsonFile -Path $resultPath -Label "$SensorName shard result"
    $failedChecks = @($result.checks.PSObject.Properties | Where-Object { -not [bool]$_.Value })
    if (
        $ClientExitCode -ne 0 -or
        [string]$result.sensor -ne $SensorName -or
        [string]$result.status -ne 'DTR_CARLA_C1_RAW_SHARD_CAPTURE_COMPLETE' -or
        $failedChecks.Count -ne 0 -or
        @($result.episodes).Count -ne 8 -or
        [int]$result.asset_count -ne 15
    ) {
        throw (
            "$SensorName shard gate failed: exit=$ClientExitCode status=$($result.status) " +
            "failed_checks=$($failedChecks.Name -join ',')"
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
    $serverStarted = $false
    $primaryFailure = $null
    $cleanupFailures = [Collections.Generic.List[string]]::new()

    try {
        Write-Output "START C1 fresh-server shard $SensorName"
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
        $serverStarted = $true
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
            $rpc = @(
                Get-CarlaListeners |
                    Where-Object { [int]$_.LocalPort -eq $script:RpcPort }
            )
            if ($rpc.Count -eq 0) {
                throw "CARLA exited during $SensorName shard capture."
            }
        }
        if (-not $clientProcess.HasExited) {
            throw "$SensorName shard exceeded $CaptureTimeoutSeconds seconds"
        }
        $clientProcess.WaitForExit()
        $clientProcess.Refresh()
        Assert-CaptureResult `
            -SensorName $SensorName `
            -ClientExitCode $clientProcess.ExitCode
        Write-Output "PASS C1 shard $SensorName"
    }
    catch {
        $primaryFailure = $_
    }
    finally {
        try {
            Stop-OwnedCapture -ClientProcess $clientProcess
        }
        catch {
            $cleanupFailures.Add($_.Exception.Message)
        }
        if ($serverStarted) {
            try {
                Stop-OwnedCarla
            }
            catch {
                $cleanupFailures.Add($_.Exception.Message)
            }
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
    Assert-CarlaIdle
}

function Invoke-Join {
    $joinStdout = Join-Path $script:LogRoot 'join.stdout.log'
    $joinStderr = Join-Path $script:LogRoot 'join.stderr.log'
    Write-Output 'START C1 cross-shard replay join and evidence seal'
    $process = Start-Process `
        -FilePath $script:CarlaPythonPath `
        -ArgumentList @(
            (Quote-ProcessArgument $script:JoinScriptPath),
            '--protocol', (Quote-ProcessArgument $script:ProtocolPath),
            '--root', (Quote-ProcessArgument $script:RawRunPath)
        ) `
        -WorkingDirectory $script:RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $joinStdout `
        -RedirectStandardError $joinStderr `
        -PassThru
    if (-not $process.WaitForExit(1200000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw 'C1 join exceeded 1200 seconds.'
    }
    $process.WaitForExit()
    $process.Refresh()
    $resultPath = Join-Path $script:RawRunPath 'result.json'
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "C1 join exited with code $($process.ExitCode) without result.json"
    }
    $result = Read-JsonFile -Path $resultPath -Label 'C1 joined result'
    $failedChecks = @($result.checks.PSObject.Properties | Where-Object { -not [bool]$_.Value })
    if (
        $process.ExitCode -ne 0 -or
        [string]$result.status -ne 'DTR_CARLA_C1_COMPLEX_SCENE_ASSET_CANARY_COMPLETE' -or
        $failedChecks.Count -ne 0
    ) {
        throw (
            "C1 joined gate failed: exit=$($process.ExitCode) status=$($result.status) " +
            "failed_checks=$($failedChecks.Name -join ',')"
        )
    }
    Write-Output 'PASS C1 cross-shard replay join and evidence seal'
}

try {
    Assert-SafeRunId -Value $RunId
    $script:CarlaLibraryRootPath = Resolve-LocalPath -Value $CarlaRoot -BasePath $script:RepoRoot
    $script:CarlaInstallRootPath = Join-Path $script:CarlaLibraryRootPath '0.9.16'
    if (-not (Test-Path -LiteralPath $script:CarlaInstallRootPath -PathType Container)) {
        throw "Packaged CARLA 0.9.16 root is unavailable: $($script:CarlaInstallRootPath)"
    }
    $script:CarlaInstallRootPath = (Resolve-Path -LiteralPath $script:CarlaInstallRootPath).Path
    $script:CarlaExePath = Join-Path $script:CarlaInstallRootPath 'CarlaUE4.exe'
    $script:CarlaPythonPath = Resolve-LocalPath -Value $CarlaPython -BasePath $script:RepoRoot
    $script:ProtocolPath = Resolve-LocalPath -Value $Protocol -BasePath $script:RepoRoot
    $script:CaptureScriptPath = Join-Path `
        $script:RepoRoot `
        'research/active/dtr-r0/carla/capture_dtr_carla_c1_complex.py'
    $script:JoinScriptPath = Join-Path `
        $script:RepoRoot `
        'research/active/dtr-r0/carla/join_dtr_carla_c1_complex.py'
    foreach ($required in @(
        @($script:CarlaExePath, 'packaged CARLA launcher'),
        @($script:CarlaPythonPath, 'CARLA client Python'),
        @($script:ProtocolPath, 'DTR-CARLA-C1 protocol'),
        @($script:CaptureScriptPath, 'DTR-CARLA-C1 capture entrypoint'),
        @($script:JoinScriptPath, 'DTR-CARLA-C1 join entrypoint')
    )) {
        Assert-RequiredFile -Path $required[0] -Label $required[1]
    }
    $protocolValue = Read-JsonFile -Path $script:ProtocolPath -Label 'DTR-CARLA-C1 protocol'
    if ([string]$protocolValue.experiment_id -ne 'DTR_CARLA_C1_COMPLEX_SCENE_ASSET_CANARY_V1') {
        throw "Unexpected C1 protocol identity: $($protocolValue.experiment_id)"
    }
    $episodeIds = @($protocolValue.scenarios | ForEach-Object { [string]$_.episode_id })
    if ($episodeIds.Count -ne 8 -or @($episodeIds | Sort-Object -Unique).Count -ne 8) {
        throw 'C1 protocol must contain exactly eight unique opaque episodes.'
    }
    $sensorOrder = @($protocolValue.capture.sensor_order | ForEach-Object { [string]$_ })
    if (($sensorOrder -join ',') -ne 'instance,wearable,depth,witness') {
        throw "C1 sensor order must be instance,wearable,depth,witness: $($sensorOrder -join ',')"
    }
    $rawRoot = Resolve-LocalPath -Value $RawEvidenceRoot -BasePath $script:RepoRoot
    $script:RawRunPath = Get-ContainedRunPath -Root $rawRoot -Child $RunId
    Assert-CarlaIdle
    [IO.Directory]::CreateDirectory($rawRoot) | Out-Null
    New-ExclusiveDirectory -Path $script:RawRunPath -Label 'raw C1 evidence run'
    $script:LogRoot = Join-Path $script:RawRunPath 'logs'
    [IO.Directory]::CreateDirectory($script:LogRoot) | Out-Null

    foreach ($sensorName in $sensorOrder) {
        Invoke-SensorCapture -SensorName $sensorName
    }
    Assert-CarlaIdle
    Invoke-Join
    Write-Output 'PASS DTR-CARLA-C1 complex scene: 4/4 shards, 8/8 episodes, 15/15 assets'
    Write-Output "evidence: $($script:RawRunPath)"
    exit 0
}
catch {
    [Console]::Error.WriteLine("DTR_CARLA_C1_RUNNER_ERROR: $($_.Exception.Message)")
    exit 2
}
