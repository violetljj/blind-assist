[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$monitor = Join-Path $PSScriptRoot "monitor_host_research_process.ps1"
$caseId = "host-monitor-test-{0}" -f ([guid]::NewGuid().ToString("N"))
$caseRelative = "artifacts.local\tmp\$caseId"
$caseDirectory = Join-Path $repoRoot $caseRelative
New-Item -ItemType Directory -Path $caseDirectory -Force | Out-Null
$parentProcess = $null

try {
    $childScript = Join-Path $caseDirectory "busy-child.ps1"
    $middleScript = Join-Path $caseDirectory "idle-child.ps1"
    $parentScript = Join-Path $caseDirectory "idle-parent.ps1"
    Set-Content -LiteralPath $childScript -Encoding UTF8 -Value @'
$end = [DateTime]::UtcNow.AddSeconds(8)
$value = 0.0
while ([DateTime]::UtcNow -lt $end) {
    $value += [Math]::Sqrt(12345.6789)
}
'@
    Set-Content -LiteralPath $middleScript -Encoding UTF8 -Value @'
param([Parameter(Mandatory = $true)][string]$BusyScript)
$info = [Diagnostics.ProcessStartInfo]::new()
$info.FileName = (Get-Command pwsh).Source
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$info.ArgumentList.Add("-NoProfile")
$info.ArgumentList.Add("-File")
$info.ArgumentList.Add($BusyScript)
$child = [Diagnostics.Process]::Start($info)
$child.WaitForExit()
exit $child.ExitCode
'@
    Set-Content -LiteralPath $parentScript -Encoding UTF8 -Value @'
param(
    [Parameter(Mandatory = $true)][string]$ChildScript,
    [Parameter(Mandatory = $true)][string]$BusyScript
)
$info = [Diagnostics.ProcessStartInfo]::new()
$info.FileName = (Get-Command pwsh).Source
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$info.ArgumentList.Add("-NoProfile")
$info.ArgumentList.Add("-File")
$info.ArgumentList.Add($ChildScript)
$info.ArgumentList.Add("-BusyScript")
$info.ArgumentList.Add($BusyScript)
$child = [Diagnostics.Process]::Start($info)
$child.WaitForExit()
exit $child.ExitCode
'@

    $parentInfo = [Diagnostics.ProcessStartInfo]::new()
    $parentInfo.FileName = (Get-Command pwsh).Source
    $parentInfo.UseShellExecute = $false
    $parentInfo.CreateNoWindow = $true
    $parentInfo.ArgumentList.Add("-NoProfile")
    $parentInfo.ArgumentList.Add("-File")
    $parentInfo.ArgumentList.Add($parentScript)
    $parentInfo.ArgumentList.Add("-ChildScript")
    $parentInfo.ArgumentList.Add($middleScript)
    $parentInfo.ArgumentList.Add("-BusyScript")
    $parentInfo.ArgumentList.Add($childScript)
    $parentProcess = [Diagnostics.Process]::Start($parentInfo)
    Start-Sleep -Milliseconds 500

    $records = @(
        & $monitor `
            -ProcessId $parentProcess.Id `
            -EvidenceDirectory $caseRelative `
            -AttemptBaseName "idle-parent-busy-child" `
            -MonitorDirectory $caseRelative `
            -PollSeconds 1 `
            -MaxSamples 4 |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
    if ($records.Count -ne 4) {
        throw "Expected 4 monitor records, got $($records.Count)."
    }
    if (-not ($records | Where-Object { $_.child_count -ge 2 })) {
        throw "Monitor did not discover the recursive descendant process."
    }
    $active = @(
        $records |
            Where-Object {
                $null -ne $_.cpu_core_equivalent -and
                [double]$_.cpu_core_equivalent -ge 0.20
            }
    )
    if ($active.Count -eq 0) {
        throw "Busy child CPU was not included in process-tree telemetry."
    }
    foreach ($record in $records) {
        foreach ($field in @(
            "gpu_utilization_percent",
            "gpu_memory_used_mib",
            "gpu_temperature_c",
            "gpu_power_draw_w"
        )) {
            if ($null -eq $record.PSObject.Properties[$field]) {
                throw "Missing optional GPU telemetry field: $field"
            }
        }
    }
    if (
        $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and
        -not (
            $records |
                Where-Object {
                    $null -ne $_.gpu_utilization_percent -and
                    $null -ne $_.gpu_memory_used_mib -and
                    $null -ne $_.gpu_temperature_c -and
                    $null -ne $_.gpu_power_draw_w
                }
        )
    ) {
        throw "Available nvidia-smi telemetry was not published."
    }
    Write-Output "PASS: host process monitor aggregates idle parent and busy child"
} finally {
    if ($null -ne $parentProcess) {
        if (-not $parentProcess.HasExited) {
            [void]$parentProcess.WaitForExit(10000)
        }
        if (-not $parentProcess.HasExited) {
            $parentProcess.Kill($true)
            $parentProcess.WaitForExit()
        }
        $parentProcess.Dispose()
    }
    $resolvedCase = [IO.Path]::GetFullPath($caseDirectory)
    $resolvedTmp = [IO.Path]::GetFullPath(
        (Join-Path $repoRoot "artifacts.local\tmp")
    )
    if ($resolvedCase.StartsWith(
        $resolvedTmp + [IO.Path]::DirectorySeparatorChar
    )) {
        Remove-Item -LiteralPath $resolvedCase -Recurse -Force
    }
}
