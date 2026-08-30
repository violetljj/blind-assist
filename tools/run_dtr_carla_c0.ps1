[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$RawEvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-c0\evidence',
    [string]$ProjectEvidenceRoot = 'artifacts.local/evidence/dtr-carla-c0',
    [string]$Model = 'artifacts.local/models/yolo11n.pt',
    [string]$Protocol = 'research/active/dtr-r0/carla/dtr_carla_c0_protocol.json',
    [string]$GpuLauncher = 'E:/codex-tools/bin/blindassist-research-gpu.cmd',
    [ValidateRange(60, 7200)]
    [int]$CaptureTimeoutSeconds = 1800
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$script:RpcPort = 2000
$script:CarlaPorts = @(2000, 2001, 2002)
$script:StartupTimeoutSeconds = 90
$script:StartupMinimumSeconds = 45
$script:CarlaHost = '127.0.0.1'
$script:StorageLeaseHelper = Join-Path $PSScriptRoot 'assert_carla_storage_capacity.ps1'
$script:StorageLeaseToken = ''
$script:StorageLeaseCarlaRoot = ''
$script:StorageLeaseOutputRoot = ''

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
    if (-not [string]::IsNullOrWhiteSpace($script:StorageLeaseToken)) {
        & $script:StorageLeaseHelper `
            -Action Release `
            -CarlaRoot $script:StorageLeaseCarlaRoot `
            -CarlaPython $script:CarlaPythonPath `
            -LeaseToken $script:StorageLeaseToken | Out-Null
        $script:StorageLeaseToken = ''
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
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing partial evidence or overwrite; $Label already exists: $Path"
    }
    try {
        New-Item -ItemType Directory -Path $Path -ErrorAction Stop | Out-Null
    }
    catch {
        throw "Unable to reserve new $Label directory '$Path': $($_.Exception.Message)"
    }
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
        throw "Refusing to share the packaged CARLA installation; process ids: $($processes.ProcessId -join ', ')"
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
        $clientStillRunning = (
            $null -ne $ClientProcess -and
            -not $ClientProcess.HasExited
        )
        if (-not $clientStillRunning -and $remaining.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)

    $remaining = @(Get-OwnedCaptureProcesses)
    $remainingIds = @($remaining.ProcessId)
    if ($null -ne $ClientProcess -and -not $ClientProcess.HasExited) {
        $remainingIds += $ClientProcess.Id
    }
    throw "Task-owned capture processes remained after cleanup: $(@($remainingIds | Sort-Object -Unique) -join ', ')"
}

function Stop-OwnedCarla {
    $owned = @(Get-CarlaInstallProcesses)
    if ($owned.Count -ne 0) {
        Stop-Process -Id $owned.ProcessId -Force -ErrorAction SilentlyContinue
    }

    # The packaged launcher and Win64-Shipping child can disappear at different
    # times. Verify the process and socket terminal state instead of assuming
    # that the first Stop-Process request completed teardown.
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
    $failures = @()
    if ($remaining.Count -ne 0) {
        $failures += "installation processes=$($remaining.ProcessId -join ', ')"
    }
    if ($listeners.Count -ne 0) {
        $failures += "listeners=$(@($listeners.LocalPort | Sort-Object -Unique) -join ', ')"
    }
    throw "CARLA cleanup verification failed: $($failures -join '; ')"
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

    $resultPath = Join-Path $script:RawRunPath "$SensorName/result.json"
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "$SensorName capture exited with code $ClientExitCode without result.json"
    }
    $result = Read-JsonFile -Path $resultPath -Label "$SensorName capture result"
    $checkProperties = @($result.checks.PSObject.Properties)
    $failedChecks = @($checkProperties | Where-Object { -not [bool]$_.Value })
    $requiredChecks = @(
        'all_scenarios_captured',
        'all_frame_counts_match',
        'all_expected_critical_match',
        'all_physical_occluders_off_route'
    )
    if ($SensorName -eq 'instance') {
        $requiredChecks += @(
            'head_yaw_changes_visibility',
            'physical_occluder_changes_visibility'
        )
    }
    $availableChecks = @($checkProperties.Name)
    $missingChecks = @($requiredChecks | Where-Object { $_ -notin $availableChecks })
    $episodeIds = @($result.episodes | ForEach-Object { [string]$_.scenario_id })
    $expectedEpisodeIds = @($script:ScenarioIds)
    $episodesMatch = (
        $episodeIds.Count -eq 12 -and
        (@($episodeIds | Sort-Object) -join "`n") -eq
            (@($expectedEpisodeIds | Sort-Object) -join "`n")
    )
    $complete = (
        $ClientExitCode -eq 0 -and
        [string]$result.sensor -eq $SensorName -and
        [string]$result.status -eq 'DTR_CARLA_C0_MODALITY_CAPTURE_COMPLETE' -and
        $missingChecks.Count -eq 0 -and
        $failedChecks.Count -eq 0 -and
        $episodesMatch
    )
    if (-not $complete) {
        $failedText = if ($failedChecks.Count -eq 0) {
            'none reported'
        }
        else {
            $failedChecks.Name -join ', '
        }
        $prefix = if ($SensorName -eq 'instance') {
            'DTR-CARLA-C0 instance gate failed; stopping before RGB capture.'
        }
        else {
            "DTR-CARLA-C0 $SensorName modality gate failed."
        }
        throw (
            "$prefix exit=$ClientExitCode status=$($result.status) " +
            "episodes=$($episodeIds.Count)/12 failed_checks=$failedText " +
            "missing_checks=$($missingChecks -join ', ')"
        )
    }
}

function Invoke-ModalityCapture {
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
        Write-Output "START modality $SensorName"
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
        $remainingStartupSeconds = (
            $script:StartupMinimumSeconds -
            [int]((Get-Date) - $serverStartedAt).TotalSeconds
        )
        if ($remainingStartupSeconds -gt 0) {
            Start-Sleep -Seconds $remainingStartupSeconds
        }
        $stableRpcListeners = @(
            Get-CarlaListeners |
                Where-Object { [int]$_.LocalPort -eq $script:RpcPort }
        )
        if ($stableRpcListeners.Count -ne 1) {
            throw "CARLA did not remain ready through the startup stabilization window"
        }

        $clientProcess = Start-Process `
            -FilePath $script:CarlaPythonPath `
            -ArgumentList @(
                (Quote-ProcessArgument $script:CaptureScriptPath),
                '--carla-root', (Quote-ProcessArgument $script:CarlaLibraryRootPath),
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

        $captureDeadline = (Get-Date).AddSeconds($CaptureTimeoutSeconds)
        while (
            -not $clientProcess.WaitForExit(1000) -and
            (Get-Date) -lt $captureDeadline
        ) {
            $rpcListeners = @(
                Get-CarlaListeners |
                    Where-Object { [int]$_.LocalPort -eq $script:RpcPort }
            )
            if ($rpcListeners.Count -eq 0) {
                throw "CARLA exited during $SensorName capture"
            }
        }
        if (-not $clientProcess.HasExited) {
            throw "$SensorName capture exceeded $CaptureTimeoutSeconds seconds"
        }
        $clientProcess.WaitForExit()
        $clientProcess.Refresh()
        Assert-CaptureResult `
            -SensorName $SensorName `
            -ClientExitCode $clientProcess.ExitCode
        Write-Output "PASS modality $SensorName (12/12 episodes)"
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
}

function Invoke-GpuStage {
    param([Parameter(Mandatory = $true)][ValidateSet('join', 'predict', 'score')][string]$Stage)

    $stageArguments = @(
        $script:PipelineScriptPath,
        $Stage,
        '--protocol', $script:ProtocolPath,
        '--root', $script:ProjectRunPath
    )
    if ($Stage -eq 'join') {
        $stageArguments += @('--raw-root', $script:RawRunPath)
    }
    elseif ($Stage -eq 'predict') {
        $stageArguments += @('--model', $script:ModelPath)
    }

    Write-Output "START GPU stage $Stage"
    Push-Location $script:RepoRoot
    try {
        & $script:GpuLauncherPath @stageArguments
        $stageExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($stageExitCode -ne 0) {
        throw "DTR-CARLA-C0 $Stage exited with code $stageExitCode"
    }
    Write-Output "PASS GPU stage $Stage"
}

try {
    Assert-SafeRunId -Value $RunId

    $script:CarlaLibraryRootPath = Resolve-LocalPath -Value $CarlaRoot -BasePath $script:RepoRoot
    if (-not (Test-Path -LiteralPath $script:CarlaLibraryRootPath -PathType Container)) {
        throw "CARLA library root is unavailable: $($script:CarlaLibraryRootPath)"
    }
    $script:CarlaLibraryRootPath = (
        Resolve-Path -LiteralPath $script:CarlaLibraryRootPath
    ).Path
    $script:CarlaInstallRootPath = Join-Path $script:CarlaLibraryRootPath '0.9.16'
    if (-not (Test-Path -LiteralPath $script:CarlaInstallRootPath -PathType Container)) {
        throw "Packaged CARLA 0.9.16 root is unavailable: $($script:CarlaInstallRootPath)"
    }
    $script:CarlaInstallRootPath = (
        Resolve-Path -LiteralPath $script:CarlaInstallRootPath
    ).Path
    $script:CarlaExePath = Join-Path $script:CarlaInstallRootPath 'CarlaUE4.exe'
    $script:CarlaPythonPath = Resolve-LocalPath -Value $CarlaPython -BasePath $script:RepoRoot
    $script:ProtocolPath = Resolve-LocalPath -Value $Protocol -BasePath $script:RepoRoot
    $script:CaptureScriptPath = Join-Path `
        $script:RepoRoot `
        'research/active/dtr-r0/carla/capture_dtr_carla_c0.py'
    $script:PipelineScriptPath = Join-Path `
        $script:RepoRoot `
        'research/active/dtr-r0/carla/dtr_carla_c0.py'
    $script:ModelPath = Resolve-LocalPath -Value $Model -BasePath $script:RepoRoot
    $script:GpuLauncherPath = Resolve-LocalPath -Value $GpuLauncher -BasePath $script:RepoRoot

    foreach ($required in @(
        @($script:CarlaExePath, 'packaged CARLA launcher'),
        @($script:CarlaPythonPath, 'CARLA client Python'),
        @($script:ProtocolPath, 'DTR-CARLA-C0 protocol'),
        @($script:CaptureScriptPath, 'DTR-CARLA-C0 capture entrypoint'),
        @($script:PipelineScriptPath, 'DTR-CARLA-C0 pipeline entrypoint'),
        @($script:ModelPath, 'DTR-CARLA-C0 model'),
        @($script:GpuLauncherPath, 'BlindAssist research GPU launcher')
    )) {
        Assert-RequiredFile -Path $required[0] -Label $required[1]
    }

    $protocolValue = Read-JsonFile `
        -Path $script:ProtocolPath `
        -Label 'DTR-CARLA-C0 protocol'
    if ([string]$protocolValue.experiment_id -ne 'DTR_CARLA_C0_CAUSAL_BENCHMARK_CANARY_V1') {
        throw "Unexpected DTR-CARLA-C0 protocol identity: $($protocolValue.experiment_id)"
    }
    $sensorOrder = @($protocolValue.capture.sensor_order | ForEach-Object { [string]$_ })
    if (($sensorOrder -join ',') -ne 'instance,rgb,depth,flow') {
        throw "Protocol sensor order is not instance,rgb,depth,flow: $($sensorOrder -join ',')"
    }
    $script:ScenarioIds = @($protocolValue.scenarios | ForEach-Object { [string]$_.id })
    if (
        $script:ScenarioIds.Count -ne 12 -or
        @($script:ScenarioIds | Sort-Object -Unique).Count -ne 12
    ) {
        throw 'DTR-CARLA-C0 protocol must contain exactly 12 unique episodes.'
    }

    $rawEvidenceRootPath = Resolve-LocalPath `
        -Value $RawEvidenceRoot `
        -BasePath $script:RepoRoot
    $projectEvidenceRootPath = Resolve-LocalPath `
        -Value $ProjectEvidenceRoot `
        -BasePath $script:RepoRoot
    $script:RawRunPath = Get-ContainedRunPath -Root $rawEvidenceRootPath -Child $RunId
    $script:ProjectRunPath = Get-ContainedRunPath `
        -Root $projectEvidenceRootPath `
        -Child $RunId

    if (Test-Path -LiteralPath $script:RawRunPath) {
        throw "Refusing partial raw evidence or overwrite: $($script:RawRunPath)"
    }
    if (Test-Path -LiteralPath $script:ProjectRunPath) {
        throw "Refusing partial project result or overwrite: $($script:ProjectRunPath)"
    }
    Assert-CarlaIdle
    $script:StorageLeaseCarlaRoot = $script:CarlaLibraryRootPath
    $script:StorageLeaseOutputRoot = $rawEvidenceRootPath
    $storageLease = & $script:StorageLeaseHelper `
        -Action Acquire `
        -CarlaRoot $script:CarlaLibraryRootPath `
        -CarlaPython $script:CarlaPythonPath `
        -ReservationBytes ([long](8GB)) `
        -OutputRoot $rawEvidenceRootPath `
        -LeaseLabel "DTR-CARLA-C0/$RunId" | ConvertFrom-Json -Depth 100
    $script:StorageLeaseToken = [string]$storageLease.lease_token

    [IO.Directory]::CreateDirectory($rawEvidenceRootPath) | Out-Null
    [IO.Directory]::CreateDirectory($projectEvidenceRootPath) | Out-Null
    New-ExclusiveDirectory -Path $script:RawRunPath -Label 'raw evidence run'
    New-ExclusiveDirectory -Path $script:ProjectRunPath -Label 'project result run'
    $script:LogRoot = Join-Path $script:RawRunPath 'logs'
    New-Item -ItemType Directory -Path $script:LogRoot -ErrorAction Stop | Out-Null

    foreach ($sensorName in $sensorOrder) {
        Invoke-ModalityCapture -SensorName $sensorName
        Assert-StorageLease
    }
    Assert-CarlaIdle

    foreach ($stage in @('join', 'predict', 'score')) {
        Invoke-GpuStage -Stage $stage
        Assert-StorageLease
    }
    foreach ($artifact in @(
        'join-result.json',
        'predictions.json',
        'prediction-receipt.json',
        'result.json',
        'result-manifest.json'
    )) {
        $artifactPath = Join-Path $script:ProjectRunPath $artifact
        Assert-RequiredFile -Path $artifactPath -Label "DTR-CARLA-C0 output $artifact"
    }

    Release-StorageLease
    Write-Output 'PASS DTR-CARLA-C0 run complete'
    Write-Output "raw_evidence: $($script:RawRunPath)"
    Write-Output "project_result: $($script:ProjectRunPath)"
    exit 0
}
catch {
    $failure = $_
    try { Release-StorageLease } catch {}
    [Console]::Error.WriteLine("DTR_CARLA_C0_RUNNER_ERROR: $($failure.Exception.Message)")
    exit 2
}
