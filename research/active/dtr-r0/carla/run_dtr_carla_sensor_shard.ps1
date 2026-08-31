[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $ProtocolPath,
    [Parameter(Mandatory)] [string] $OutputRoot,
    [Parameter(Mandatory)]
    [ValidateSet('instance', 'wearable', 'depth', 'witness')]
    [string] $Sensor,
    [Parameter(Mandatory)] [string] $WorkRoot,
    [ValidateRange(1024, 65533)] [int] $Port = 26400,
    [ValidateRange(30, 900)] [int] $RpcTimeoutSeconds = 120,
    [ValidateRange(300, 7200)] [int] $CaptureTimeoutSeconds = 2400
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$repo = 'E:\linnan\linnan'
$carlaRoot = 'E:\linnan\CARLA'
$carlaInstall = 'E:\linnan\CARLA\0.9.16'
$carlaExe = Join-Path $carlaInstall 'CarlaUE4.exe'
$python = 'E:\linnan\CARLA\client-env\Scripts\python.exe'
$captureScript = Join-Path $repo 'research\active\dtr-r0\carla\capture_dtr_carla_c2_rich_scene.py'
$storageHelper = Join-Path $repo 'tools\assert_carla_storage_capacity.ps1'
$protocol = [IO.Path]::GetFullPath($ProtocolPath)
$output = [IO.Path]::GetFullPath($OutputRoot)
$work = [IO.Path]::GetFullPath($WorkRoot)
$portGroup = @($Port, $Port + 1, $Port + 2)
$server = $null
$client = $null
$leaseToken = ''

function Get-OwnedCarlaProcesses {
    @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                [string]$_.Name -match '^CarlaUE4' -and
                [string]$_.CommandLine -match "-carla-rpc-port=$Port(?:\s|$)"
            }
    )
}

function Get-ShardPortListeners {
    @(
        Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
            Where-Object { [int]$_.LocalPort -in $portGroup }
    )
}

foreach ($required in @($carlaExe, $python, $captureScript, $storageHelper, $protocol)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required shard dependency is unavailable: $required"
    }
}
if (-not (Test-Path -LiteralPath $output -PathType Container)) {
    throw "Output root must already exist: $output"
}
if (Test-Path -LiteralPath (Join-Path $output "shards\$Sensor")) {
    throw "Refusing shard overwrite: $Sensor"
}
if (@(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            [string]$_.Name -match '^CarlaUE4'
        }).Count -ne 0) {
    throw 'Another CARLA allocation is active.'
}
if (@(Get-ShardPortListeners).Count -ne 0) {
    throw "CARLA ports are already in use: $($portGroup -join ',')"
}
New-Item -ItemType Directory -Path $work -Force | Out-Null

$engineIni = Join-Path $work 'Town10HD_Opt.Engine.ini'
$serverStdout = Join-Path $work 'server.stdout.log'
$serverStderr = Join-Path $work 'server.stderr.log'
$clientStdout = Join-Path $work 'client.stdout.log'
$clientStderr = Join-Path $work 'client.stderr.log'
$engineText = @(
    '[/Script/EngineSettings.GameMapsSettings]'
    'GameDefaultMap=/Game/Carla/Maps/Town10HD_Opt.Town10HD_Opt'
    'ServerDefaultMap=/Game/Carla/Maps/Town10HD_Opt.Town10HD_Opt'
    'TransitionMap=/Game/Carla/Maps/Town10HD_Opt.Town10HD_Opt'
    'GlobalDefaultGameMode=/Game/Carla/Blueprints/Game/CarlaGameMode.CarlaGameMode_C'
    'GlobalDefaultServerGameMode=/Game/Carla/Blueprints/Game/CarlaGameMode.CarlaGameMode_C'
    ''
) -join "`n"
[IO.File]::WriteAllText($engineIni, $engineText, [Text.UTF8Encoding]::new($false))

try {
    $lease = & $storageHelper `
        -Action Acquire `
        -CarlaRoot $carlaRoot `
        -CarlaPython $python `
        -ExperimentsRoot (Split-Path -Parent $output) `
        -ReservationBytes ([long](2GB)) `
        -OutputRoot $output `
        -LeaseLabel "DTR-CARLA-$Sensor" `
        -OwnerPid $PID | ConvertFrom-Json -Depth 100
    $leaseToken = [string]$lease.lease_token

    $serverStarted = Get-Date
    $server = Start-Process `
        -FilePath $carlaExe `
        -ArgumentList @(
            "-EngineIni=$engineIni", '-dx12', '-RenderOffScreen', '-nosound',
            '-quality-level=Low', "-carla-rpc-port=$Port"
        ) `
        -WorkingDirectory $carlaInstall `
        -WindowStyle Hidden `
        -RedirectStandardOutput $serverStdout `
        -RedirectStandardError $serverStderr `
        -PassThru

    $startupDeadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $startupDeadline) {
        if (@(Get-ShardPortListeners | Where-Object LocalPort -eq $Port).Count -ne 0) {
            break
        }
        $server.Refresh()
        if ($server.HasExited) {
            throw 'CARLA exited before opening the RPC port.'
        }
        Start-Sleep -Seconds 1
    }
    if (@(Get-ShardPortListeners | Where-Object LocalPort -eq $Port).Count -eq 0) {
        throw 'CARLA startup timed out.'
    }
    $remainingWarmup = 45 - [int]((Get-Date) - $serverStarted).TotalSeconds
    if ($remainingWarmup -gt 0) {
        Start-Sleep -Seconds $remainingWarmup
    }

    $client = Start-Process `
        -FilePath $python `
        -ArgumentList @(
            $captureScript, '--protocol', $protocol, '--output-root', $output,
            '--sensor', $Sensor, '--host', '127.0.0.1', '--port', [string]$Port,
            '--rpc-timeout-seconds', [string]$RpcTimeoutSeconds
        ) `
        -WorkingDirectory $repo `
        -WindowStyle Hidden `
        -RedirectStandardOutput $clientStdout `
        -RedirectStandardError $clientStderr `
        -PassThru

    $deadline = (Get-Date).AddSeconds($CaptureTimeoutSeconds)
    while (-not $client.WaitForExit(1000) -and (Get-Date) -lt $deadline) {
        if (@(Get-OwnedCarlaProcesses).Count -eq 0) {
            throw 'Task-owned CARLA process disappeared during capture.'
        }
    }
    if (-not $client.HasExited) {
        throw "$Sensor capture exceeded $CaptureTimeoutSeconds seconds."
    }
    $client.WaitForExit()
    $client.Refresh()
    if ($client.ExitCode -ne 0) {
        throw "$Sensor capture failed with exit code $($client.ExitCode)."
    }
    $resultPath = Join-Path $output "shards\$Sensor\result.json"
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "$Sensor capture produced no result."
    }
    Get-Content -LiteralPath $resultPath -Raw
}
finally {
    if ($null -ne $client -and -not $client.HasExited) {
        Stop-Process -Id $client.Id -Force -ErrorAction SilentlyContinue
    }
    $owned = @(Get-OwnedCarlaProcesses)
    if ($owned.Count -ne 0) {
        Stop-Process -Id $owned.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $cleanupDeadline = (Get-Date).AddSeconds(30)
    while ((@(Get-OwnedCarlaProcesses).Count -ne 0 -or @(Get-ShardPortListeners).Count -ne 0) -and (Get-Date) -lt $cleanupDeadline) {
        Start-Sleep -Milliseconds 500
    }
    if (@(Get-OwnedCarlaProcesses).Count -ne 0 -or @(Get-ShardPortListeners).Count -ne 0) {
        throw 'Task-owned CARLA cleanup verification failed.'
    }
    if (-not [string]::IsNullOrWhiteSpace($leaseToken)) {
        & $storageHelper `
            -Action Release `
            -CarlaRoot $carlaRoot `
            -CarlaPython $python `
            -ExperimentsRoot (Split-Path -Parent $output) `
            -LeaseToken $leaseToken `
            -OwnerPid $PID | Out-Null
    }
}
