[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [string]$Plan,

    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$RawEvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-n1-natural-dynamics\evidence',
    [string]$CaptureScript = 'research/active/dtr-r0/carla/capture_dtr_carla_n1_natural_dynamics.py',
    [string]$StartupEngineIni = '',
    [ValidateRange(1024, 65529)]
    [int]$RpcPort = 26000,
    [ValidateRange(1024, 65535)]
    [int]$TrafficManagerPort = 26003,
    [ValidateRange(120, 3600)]
    [int]$CaptureTimeoutSeconds = 900,
    [ValidateRange(2.0, 16.0)]
    [double]$MinimumFreePhysicalGB = 4.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$carlaInstallRoot = [IO.Path]::GetFullPath((Join-Path $CarlaRoot '0.9.16'))
$carlaExe = Join-Path $carlaInstallRoot 'CarlaUE4.exe'
$carlaPythonPath = [IO.Path]::GetFullPath($CarlaPython)
$planPath = if ([IO.Path]::IsPathRooted($Plan)) {
    [IO.Path]::GetFullPath($Plan)
} else {
    [IO.Path]::GetFullPath((Join-Path $repoRoot $Plan))
}
$capturePath = if ([IO.Path]::IsPathRooted($CaptureScript)) {
    [IO.Path]::GetFullPath($CaptureScript)
} else {
    [IO.Path]::GetFullPath((Join-Path $repoRoot $CaptureScript))
}
$startupEngineIniPath = if ([string]::IsNullOrWhiteSpace($StartupEngineIni)) {
    ''
} elseif ([IO.Path]::IsPathRooted($StartupEngineIni)) {
    [IO.Path]::GetFullPath($StartupEngineIni)
} else {
    [IO.Path]::GetFullPath((Join-Path $repoRoot $StartupEngineIni))
}
$evidenceRoot = [IO.Path]::GetFullPath($RawEvidenceRoot).TrimEnd('\', '/')
$runRoot = [IO.Path]::GetFullPath((Join-Path $evidenceRoot $RunId))
$expectedRunPrefix = $evidenceRoot + [IO.Path]::DirectorySeparatorChar
$ports = @($RpcPort, $RpcPort + 1, $RpcPort + 2, $TrafficManagerPort)

if (-not $runRoot.StartsWith($expectedRunPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Run path escapes evidence root: $runRoot"
}
foreach ($required in @($carlaExe, $carlaPythonPath, $planPath, $capturePath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is unavailable: $required"
    }
}
if (
    -not [string]::IsNullOrWhiteSpace($startupEngineIniPath) -and
    -not (Test-Path -LiteralPath $startupEngineIniPath -PathType Leaf)
) {
    throw "Startup Engine.ini is unavailable: $startupEngineIniPath"
}
$planDocument = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json -Depth 100
$protocolMap = [string]$planDocument.environment.map
if (Test-Path -LiteralPath $runRoot) {
    throw "Refusing evidence overwrite: $runRoot"
}

function Get-PortListeners {
    @(
        Get-NetTCPConnection `
            -LocalPort $ports `
            -State Listen `
            -ErrorAction SilentlyContinue
    )
}

function Get-TaskCarlaProcesses {
    $prefix = $carlaInstallRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $portArgument = "-carla-rpc-port=$RpcPort"
    @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
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
    $prefix = $carlaInstallRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $executable = [string]$_.ExecutablePath
                -not [string]::IsNullOrWhiteSpace($executable) -and
                $executable.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
            }
    )
}

function Stop-TaskResources {
    param([AllowNull()][System.Diagnostics.Process]$ClientProcess)
    if ($null -ne $ClientProcess -and -not $ClientProcess.HasExited) {
        Stop-Process -Id $ClientProcess.Id -Force -ErrorAction SilentlyContinue
    }
    $deadline = (Get-Date).AddSeconds(30)
    do {
        $owned = @(Get-TaskCarlaProcesses)
        if ($owned.Count -ne 0) {
            Stop-Process -Id $owned.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 500
        $listeners = @(Get-PortListeners)
    } while (($owned.Count -ne 0 -or $listeners.Count -ne 0) -and (Get-Date) -lt $deadline)
    $owned = @(Get-TaskCarlaProcesses)
    $listeners = @(Get-PortListeners)
    if ($owned.Count -ne 0 -or $listeners.Count -ne 0) {
        throw (
            'N1 cleanup verification failed: processes=' +
            ($owned.ProcessId -join ',') + '; listeners=' +
            ($listeners.LocalPort -join ',')
        )
    }
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
    $expected = Get-ExpectedEngineMapObjectPath -MapName $MapName
    $content = Get-Content -LiteralPath $Path -Raw
    foreach ($key in @('GameDefaultMap', 'ServerDefaultMap', 'TransitionMap')) {
        $matches = @([regex]::Matches(
            $content,
            "(?m)^$([regex]::Escape($key))=(?<value>[^\r\n]+)$"
        ))
        if ($matches.Count -ne 1 -or [string]$matches[0].Groups['value'].Value -ne $expected) {
            throw "Startup Engine.ini does not bind $key to $expected"
        }
    }
}

$existingListeners = @(Get-PortListeners)
if ($existingListeners.Count -ne 0) {
    throw "CARLA/TM ports are already in use: $($existingListeners.LocalPort -join ',')"
}
$existingCarla = @(Get-AllCarlaProcesses)
if ($existingCarla.Count -ne 0) {
    throw "Another CARLA allocation is active: $($existingCarla.ProcessId -join ',')"
}
$os = Get-CimInstance Win32_OperatingSystem
$freePhysicalGB = [double]$os.FreePhysicalMemory / 1MB
if ($freePhysicalGB -lt $MinimumFreePhysicalGB) {
    throw "Insufficient free physical memory for N1: $([Math]::Round($freePhysicalGB, 2)) GiB"
}

[IO.Directory]::CreateDirectory($runRoot) | Out-Null
$logRoot = Join-Path $runRoot 'logs'
[IO.Directory]::CreateDirectory($logRoot) | Out-Null
$frozenPlan = Join-Path $runRoot 'frozen_plan.json'
Copy-Item -LiteralPath $planPath -Destination $frozenPlan
$planHash = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash
if ((Get-FileHash -LiteralPath $frozenPlan -Algorithm SHA256).Hash -ne $planHash) {
    throw 'Frozen N1 plan differs from its source.'
}

$server = $null
$clientProcess = $null
$runtimeStartupEngineIniPath = ''
$primaryFailure = $null
$cleanupFailure = $null
try {
    $serverArguments = [Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($startupEngineIniPath)) {
        Assert-StartupEngineIniMap -Path $startupEngineIniPath -MapName $protocolMap
        $runtimeStartupEngineIniPath = Join-Path (
            [IO.Path]::GetTempPath()
        ) "blindassist-n1-$PID-$RpcPort-$([Guid]::NewGuid().ToString('N')).Engine.ini"
        Copy-Item -LiteralPath $startupEngineIniPath -Destination $runtimeStartupEngineIniPath
        Assert-StartupEngineIniMap -Path $runtimeStartupEngineIniPath -MapName $protocolMap
        $serverArguments.Add("-EngineIni=$runtimeStartupEngineIniPath")
    }
    foreach ($argument in @(
        '-dx12',
        '-RenderOffScreen',
        '-nosound',
        '-quality-level=Epic',
        "-carla-rpc-port=$RpcPort"
    )) {
        $serverArguments.Add($argument)
    }
    $server = Start-Process `
        -FilePath $carlaExe `
        -ArgumentList $serverArguments.ToArray() `
        -WorkingDirectory $carlaInstallRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot 'server.stdout.log') `
        -RedirectStandardError (Join-Path $logRoot 'server.stderr.log') `
        -PassThru

    $startupDeadline = (Get-Date).AddSeconds(180)
    do {
        Start-Sleep -Seconds 1
        $rpcReady = @(
            Get-PortListeners | Where-Object { [int]$_.LocalPort -eq $RpcPort }
        )
        $server.Refresh()
        if ($server.HasExited) {
            throw 'CARLA N1 server exited before the RPC port opened.'
        }
    } while ($rpcReady.Count -eq 0 -and (Get-Date) -lt $startupDeadline)
    if ($rpcReady.Count -eq 0) {
        throw 'CARLA N1 startup timed out.'
    }
    Start-Sleep -Seconds 20

    $clientProcess = Start-Process `
        -FilePath $carlaPythonPath `
        -ArgumentList @(
            (Quote-ProcessArgument $capturePath),
            '--plan', (Quote-ProcessArgument $frozenPlan),
            '--output-root', (Quote-ProcessArgument $runRoot),
            '--host', '127.0.0.1',
            '--port', [string]$RpcPort,
            '--traffic-manager-port', [string]$TrafficManagerPort
        ) `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot 'client.stdout.log') `
        -RedirectStandardError (Join-Path $logRoot 'client.stderr.log') `
        -PassThru

    $captureDeadline = (Get-Date).AddSeconds($CaptureTimeoutSeconds)
    while (-not $clientProcess.WaitForExit(1000) -and (Get-Date) -lt $captureDeadline) {
        $rpcReady = @(
            Get-PortListeners | Where-Object { [int]$_.LocalPort -eq $RpcPort }
        )
        if ($rpcReady.Count -eq 0) {
            throw 'CARLA exited during the N1 materialization.'
        }
    }
    if (-not $clientProcess.HasExited) {
        throw "N1 materialization exceeded $CaptureTimeoutSeconds seconds."
    }
    $clientProcess.WaitForExit()
    $clientProcess.Refresh()
    $resultPath = Join-Path $runRoot 'result.json'
    if ($clientProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "N1 materialization failed with exit code $($clientProcess.ExitCode)."
    }
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json -Depth 100
    $failedChecks = @(
        $result.checks.PSObject.Properties |
            Where-Object { $_.Value -isnot [bool] -or $_.Value -ne $true }
    )
    if (
        [string]$result.status -ne 'DTR_CARLA_N1_NATURAL_DYNAMICS_MATERIALIZED' -or
        $failedChecks.Count -ne 0
    ) {
        throw "N1 result gate failed: $($failedChecks.Name -join ',')"
    }
    Write-Output "PASS N1 natural dynamics: $resultPath"
}
catch {
    $primaryFailure = $_
}
finally {
    try {
        Stop-TaskResources -ClientProcess $clientProcess
    }
    catch {
        $cleanupFailure = $_
    }
    if (
        -not [string]::IsNullOrWhiteSpace($runtimeStartupEngineIniPath) -and
        (Test-Path -LiteralPath $runtimeStartupEngineIniPath -PathType Leaf)
    ) {
        Remove-Item -LiteralPath $runtimeStartupEngineIniPath -Force
    }
}

if ($null -ne $cleanupFailure) {
    if ($null -ne $primaryFailure) {
        throw "$($primaryFailure.Exception.Message) Cleanup also failed: $($cleanupFailure.Exception.Message)"
    }
    throw $cleanupFailure
}
if ($null -ne $primaryFailure) {
    throw $primaryFailure
}
