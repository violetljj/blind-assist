[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')]
    [string]$RunId,

    [string]$SourceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-n1-natural-dynamics\evidence\n1-native-pilot-v4-20260830-2345',
    [string]$CarlaRoot = 'E:\linnan\CARLA',
    [string]$CarlaPython = 'E:\linnan\CARLA\client-env\Scripts\python.exe',
    [string]$EvidenceRoot = 'E:\linnan\CARLA\experiments\dtr-carla-n2-frozen-trace-replay\evidence',
    [string]$Protocol = 'research/active/dtr-r0/carla/dtr_carla_n2_frozen_trace_replay_protocol.json',
    [string]$HelperScript = 'research/active/dtr-r0/carla/dtr_carla_n2_frozen_trace_replay.py',
    [string]$CaptureScript = 'research/active/dtr-r0/carla/capture_dtr_carla_n2_frozen_trace_replay.py',
    [string]$StartupEngineIni = '',
    [ValidateRange(1024, 65533)]
    [int]$RpcPort = 26100,
    [ValidateRange(120, 3600)]
    [int]$CaptureTimeoutSeconds = 1800,
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

function Resolve-InputPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

$sourcePath = Resolve-InputPath $SourceRoot
$protocolPath = Resolve-InputPath $Protocol
$helperPath = Resolve-InputPath $HelperScript
$capturePath = Resolve-InputPath $CaptureScript
$startupEngineIniPath = if ([string]::IsNullOrWhiteSpace($StartupEngineIni)) {
    ''
} else {
    Resolve-InputPath $StartupEngineIni
}
$evidenceRootPath = [IO.Path]::GetFullPath($EvidenceRoot).TrimEnd('\', '/')
$runRoot = [IO.Path]::GetFullPath((Join-Path $evidenceRootPath $RunId))
$expectedRunPrefix = $evidenceRootPath + [IO.Path]::DirectorySeparatorChar
$ports = @($RpcPort, $RpcPort + 1, $RpcPort + 2)

if (-not $runRoot.StartsWith($expectedRunPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Run path escapes N2 evidence root: $runRoot"
}
foreach ($required in @($carlaExe, $carlaPythonPath, $protocolPath, $helperPath, $capturePath)) {
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
$protocolDocument = Get-Content -LiteralPath $protocolPath -Raw | ConvertFrom-Json -Depth 100
$protocolMap = [string]$protocolDocument.environment.map
if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
    throw "Frozen N1 source root is unavailable: $sourcePath"
}
if (Test-Path -LiteralPath $runRoot) {
    throw "Refusing N2 evidence overwrite: $runRoot"
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

function Get-PortListeners {
    @(
        Get-NetTCPConnection -LocalPort $ports -State Listen -ErrorAction SilentlyContinue
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
            'N2 cleanup verification failed: processes=' +
            ($owned.ProcessId -join ',') + '; listeners=' +
            ($listeners.LocalPort -join ',')
        )
    }
}

$preflightOutput = & $carlaPythonPath -B $helperPath `
    --protocol $protocolPath `
    --source-root $sourcePath 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Frozen N1 source preflight failed: $($preflightOutput -join [Environment]::NewLine)"
}

$existingListeners = @(Get-PortListeners)
if ($existingListeners.Count -ne 0) {
    throw "N2 CARLA ports are already in use: $($existingListeners.LocalPort -join ',')"
}
$existingCarla = @(Get-AllCarlaProcesses)
if ($existingCarla.Count -ne 0) {
    throw "Another CARLA allocation is active: $($existingCarla.ProcessId -join ',')"
}
$os = Get-CimInstance Win32_OperatingSystem
$freePhysicalGB = [double]$os.FreePhysicalMemory / 1MB
if ($freePhysicalGB -lt $MinimumFreePhysicalGB) {
    throw "Insufficient free physical memory for N2: $([Math]::Round($freePhysicalGB, 2)) GiB"
}

[IO.Directory]::CreateDirectory($runRoot) | Out-Null
$inputRoot = Join-Path $runRoot 'inputs'
$logRoot = Join-Path $runRoot 'logs'
[IO.Directory]::CreateDirectory($inputRoot) | Out-Null
[IO.Directory]::CreateDirectory($logRoot) | Out-Null
$frozenProtocol = Join-Path $runRoot 'frozen_protocol.json'
Copy-Item -LiteralPath $protocolPath -Destination $frozenProtocol
foreach ($name in @(
    'behavior_trace.jsonl',
    'actor_manifest.json',
    'frozen_plan.json',
    'event_receipts.json',
    'result.json'
)) {
    Copy-Item -LiteralPath (Join-Path $sourcePath $name) -Destination (Join-Path $inputRoot $name)
}

$copiedPreflightOutput = & $carlaPythonPath -B $helperPath `
    --protocol $frozenProtocol `
    --source-root $inputRoot `
    --receipt (Join-Path $runRoot 'preflight_source_bundle_receipt.json') 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Copied N2 source preflight failed: $($copiedPreflightOutput -join [Environment]::NewLine)"
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
        ) "blindassist-n2-$PID-$RpcPort-$([Guid]::NewGuid().ToString('N')).Engine.ini"
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
            throw 'CARLA N2 server exited before the RPC port opened.'
        }
    } while ($rpcReady.Count -eq 0 -and (Get-Date) -lt $startupDeadline)
    if ($rpcReady.Count -eq 0) {
        throw 'CARLA N2 startup timed out.'
    }
    Start-Sleep -Seconds 20

    $clientProcess = Start-Process `
        -FilePath $carlaPythonPath `
        -ArgumentList @(
            '-B',
            (Quote-ProcessArgument $capturePath),
            '--protocol', (Quote-ProcessArgument $frozenProtocol),
            '--source-root', (Quote-ProcessArgument $inputRoot),
            '--output-root', (Quote-ProcessArgument $runRoot),
            '--host', '127.0.0.1',
            '--port', [string]$RpcPort
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
            throw 'CARLA exited during the N2 replay.'
        }
    }
    if (-not $clientProcess.HasExited) {
        throw "N2 replay exceeded $CaptureTimeoutSeconds seconds."
    }
    $clientProcess.WaitForExit()
    $clientProcess.Refresh()
    $resultPath = Join-Path $runRoot 'result.json'
    if ($clientProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "N2 replay failed with exit code $($clientProcess.ExitCode)."
    }
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json -Depth 100
    $failedChecks = @(
        $result.checks.PSObject.Properties |
            Where-Object { $_.Value -isnot [bool] -or $_.Value -ne $true }
    )
    if (
        [string]$result.status -ne 'DTR_CARLA_N2_FROZEN_TRACE_C2_REPLAY_COMPLETE' -or
        $failedChecks.Count -ne 0
    ) {
        throw "N2 replay gate failed: $($failedChecks.Name -join ',')"
    }
    Write-Output "PASS N2 frozen trace C2 replay: $resultPath"
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
