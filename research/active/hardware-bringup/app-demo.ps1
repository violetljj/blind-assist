#requires -Version 7.0
<#
USB hardware -> existing PC dashboard -> ADB reverse -> Android App demo.
Start retains its forwarding for the presentation; recording ends after Seconds.
Use -Action Stop afterward to release only receipt-owned resources.
Check is read-only. This launcher never starts/stops the dashboard server.
#>
param(
    [ValidateSet('Start', 'Stop', 'Check')][string]$Action = 'Check',
    [string]$Serial,
    [ValidatePattern('^COM[0-9]+$')][string]$CameraPort = 'COM4',
    [ValidatePattern('^COM[0-9]+$')][string]$TofPort = 'COM5',
    [ValidateRange(3, 300)][int]$Seconds = 120,
    [string]$AdbPath
)
$ErrorActionPreference = 'Stop'
$baseUrl = 'http://127.0.0.1:8766'
$reverseEndpoint = 'tcp:8766'
$cameraExplicit = $PSBoundParameters.ContainsKey('CameraPort')
$tofExplicit = $PSBoundParameters.ContainsKey('TofPort')
if (-not $AdbPath) {
    $found = Get-Command adb.exe -ErrorAction SilentlyContinue
    if ($found) { $AdbPath = $found.Source }
    else {
        $AdbPath = @(
            'E:/codex-tools/tools/android-sdk/platform-tools/adb.exe',
            (Join-Path $env:LOCALAPPDATA 'Android/Sdk/platform-tools/adb.exe'),
            'E:/codex-tools/projects/blindassist/toolchain/android-sdk/platform-tools/adb.exe'
        ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    }
}
if (-not $AdbPath -or -not (Test-Path -LiteralPath $AdbPath -PathType Leaf)) {
    throw 'ADB executable missing; pass -AdbPath <adb.exe>.'
}
function Invoke-Adb([string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new($AdbPath)
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(15000)) {
            $process.Kill()
            throw 'ADB command timed out after 15 seconds.'
        }
        $output = $stdout.GetAwaiter().GetResult().Trim()
        $errorText = $stderr.GetAwaiter().GetResult().Trim()
        if ($process.ExitCode -ne 0) { throw "ADB failed: $errorText $output" }
        return $output
    } finally { $process.Dispose() }
}
function Get-DashboardState {
    $state = Invoke-RestMethod "$baseUrl/api/state" -TimeoutSec 5
    if ($state.mode -notin @('idle', 'live', 'replay') -or -not $state.limits) {
        throw 'Local port 8766 does not identify the hardware dashboard.'
    }
    return $state
}
function Get-ReverseMapping {
    $rows = @((Invoke-Adb @('-s', $Serial, 'reverse', '--list')) -split '\r?\n' |
        Where-Object { $_.Trim() } | ForEach-Object { ,($_.Trim() -split '\s+') })
    $matchesForPort = @($rows | Where-Object { $_.Count -ge 3 -and $_[1] -eq $reverseEndpoint })
    if ($matchesForPort.Count -gt 1) { throw 'Ambiguous ADB reverse listing; no mapping changed.' }
    if ($matchesForPort.Count -eq 1) { return ,$matchesForPort[0] }
    return $null
}
function Save-Receipt {
    $receipt.updated_utc = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding utf8
}
function Resolve-HardwarePort([string]$Requested, [string]$Role, [bool]$Explicit) {
    $matching = @($ports | Where-Object {
        $_.port -eq $Requested -and $_.vid -eq 0x303A -and $_.pid -eq 0x1001 -and
        $_.serial_number -and $_.role_hint -eq $Role
    })
    if ($matching.Count -eq 1) { return $matching[0].port }
    if (-not $Explicit) {
        $candidates = @($ports | Where-Object {
            $_.vid -eq 0x303A -and $_.pid -eq 0x1001 -and $_.serial_number -and $_.role_hint -eq $Role
        })
        if ($candidates.Count -eq 1) { return $candidates[0].port }
    }
    throw "Cannot verify $Requested as $Role using local USB metadata and dashboard role_hint. No capture started."
}
function Remove-OwnedMapping {
    if (-not $receipt.mapping_owned) { return }
    $current = Get-ReverseMapping
    if ($current -and $current[2] -ne $reverseEndpoint) {
        throw 'Owned mapping was replaced with a different target; preserving it.'
    }
    if ($current) { [void](Invoke-Adb @('-s', $Serial, 'reverse', '--remove', $reverseEndpoint)) }
    if (Get-ReverseMapping) { throw 'ADB forwarding removal could not be verified.' }
    $receipt.mapping_owned = $false
    Save-Receipt
}
function Stop-OwnedRecording {
    if (-not $receipt.recording_owned -or -not $receipt.run_id) { return }
    $current = Get-DashboardState
    if ($current.run_id -eq $receipt.run_id -and $current.recording.active) {
        # Dashboard has no conditional-stop API. Recheck ownership immediately before
        # stopping, and never stop an already-observed different user's run.
        $current = Get-DashboardState
        if ($current.run_id -eq $receipt.run_id -and $current.recording.active) {
            [void](Invoke-RestMethod "$baseUrl/api/stop" -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 5)
            $deadline = [DateTime]::UtcNow.AddSeconds(15)
            do {
                Start-Sleep -Milliseconds 250
                $current = Get-DashboardState
            } while ($current.run_id -eq $receipt.run_id -and $current.recording.active -and [DateTime]::UtcNow -lt $deadline)
            if ($current.run_id -eq $receipt.run_id -and $current.recording.active) {
                throw 'Owned recording still finalizing; its receipt is retained for retry.'
            }
        }
    }
    $receipt.recording_owned = $false
    Save-Receipt
}

$devices = @(foreach ($line in ((Invoke-Adb @('devices')) -split '\r?\n')) {
    if ($line -match '^(\S+)\s+device\s*$') { $Matches[1] }
})
if (-not $Serial) {
    if ($devices.Count -ne 1) { throw 'Require one ready Android device, or explicitly select -Serial.' }
    $Serial = $devices[0]
}
if ($Serial -notmatch '^[A-Za-z0-9._:-]+$' -or $Serial -notin $devices) {
    throw 'Selected Android device is not authorized and ready.'
}
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$receiptRoot = Join-Path $repoRoot 'artifacts.local/hardware-bringup/app-demo'
$receiptPath = Join-Path $receiptRoot (($Serial -replace ':', '_') + '-8766.json')
$receipt = if (Test-Path -LiteralPath $receiptPath) {
    Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json -AsHashtable
} else { $null }
if ($receipt -and ($receipt.serial -ne $Serial -or $receipt.port -ne 8766 -or $receipt.owner -ne 'app-demo.ps1')) {
    throw 'Receipt identity mismatch; no resource changed.'
}

if ($Action -eq 'Stop') {
    if (-not $receipt) { throw 'No app-demo ownership receipt; no resource changed.' }
    $errors = @()
    try { Stop-OwnedRecording } catch { $errors += $_.Exception.Message }
    try { Remove-OwnedMapping } catch { $errors += $_.Exception.Message }
    if ($errors.Count) { throw ($errors -join '; ') }
    [pscustomobject]@{ status = 'stopped_owned_resources'; receipt = $receiptPath } | ConvertTo-Json
    exit 0
}

$state = Get-DashboardState
$ports = Invoke-RestMethod "$baseUrl/api/ports" -TimeoutSec 5
$selectedCamera = Resolve-HardwarePort $CameraPort 'camera' $cameraExplicit
$selectedTof = Resolve-HardwarePort $TofPort 'tof' $tofExplicit
if ($selectedCamera -eq $selectedTof) { throw 'Camera and ToF must use different verified ports.' }
$mapping = Get-ReverseMapping
if ($mapping -and $mapping[2] -ne $reverseEndpoint) {
    throw 'Phone port 8766 already forwards elsewhere; preserving existing mapping.'
}
if ($Action -eq 'Check') {
    [pscustomobject]@{
        status = 'ready'; serial = $Serial; camera_port = $selectedCamera; tof_port = $selectedTof
        mode = $state.mode; recording_active = [bool]$state.recording.active
        forwarded = [bool]$mapping; receipt_exists = [bool]$receipt
    } | ConvertTo-Json
    exit 0
}

[void](New-Item -ItemType Directory -Path $receiptRoot -Force)
if (-not $receipt) {
    $receipt = @{
        owner = 'app-demo.ps1'; serial = $Serial; port = 8766; mapping_owned = $false
        recording_owned = $false; run_id = $null; updated_utc = $null
    }
}
$createdMapping = $false
$createdRun = $false
$succeeded = $false
try {
    if (-not $mapping) {
        [void](Invoke-Adb @('-s', $Serial, 'reverse', '--no-rebind', $reverseEndpoint, $reverseEndpoint))
        $createdMapping = $true
        $receipt.mapping_owned = $true
        Save-Receipt
    }
    $state = Get-DashboardState
    if (-not $state.recording.active) {
        $started = Invoke-RestMethod "$baseUrl/api/start" -Method Post -ContentType 'application/json' -Body (
            @{ camera_port = $selectedCamera; tof_port = $selectedTof; seconds = $Seconds } | ConvertTo-Json
        ) -TimeoutSec 10
        if (-not $started.run_id) { throw 'Dashboard did not acknowledge a run_id; inspect its bounded recording.' }
        $createdRun = $true
        $receipt.run_id = $started.run_id
        $receipt.recording_owned = $true
        $receipt.seconds = $Seconds
        $receipt.camera_port = $selectedCamera
        $receipt.tof_port = $selectedTof
        Save-Receipt
    } elseif ($receipt.run_id -ne $state.run_id) {
        # An already-active user recording is only displayed, never taken over.
        $receipt.recording_owned = $false
        $receipt.run_id = $null
        Save-Receipt
    }
    $launch = Invoke-Adb @('-s', $Serial, 'shell', 'am', 'start', '-W', '-n', 'com.linnan.blindassist/.MainActivity')
    if ($launch -match '(?im)^Error:|Exception') { throw $launch }
    $succeeded = $true
    [pscustomobject]@{
        status = 'app_launch_requested'; serial = $Serial; camera_port = $selectedCamera; tof_port = $selectedTof
        created_recording = $createdRun; requested_seconds = if ($createdRun) { $Seconds } else { $null }
        run_id = if ($createdRun) { $receipt.run_id } else { $state.run_id }; receipt = $receiptPath
        next = '在 App 设备中心点击硬件展示。录制到时自动结束；展示结束后运行本脚本 -Action Stop 释放自建转发。'
        launch = $launch
    } | ConvertTo-Json
} finally {
    if (-not $succeeded) {
        # Roll back only acquisitions of this invocation, never older/user resources.
        if ($createdRun) {
            try { Stop-OwnedRecording } catch { Write-Warning "Recording cleanup incomplete: $($_.Exception.Message)" }
        }
        if ($createdMapping) {
            try { Remove-OwnedMapping } catch { Write-Warning "Forwarding cleanup incomplete: $($_.Exception.Message)" }
        }
    }
}
