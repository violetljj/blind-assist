[CmdletBinding()]
param(
    [ValidateSet('Inspect', 'Exec')][string]$Action = 'Inspect',
    [string]$ConfigPath,
    [string]$ScriptFile,
    [ValidateSet('research','l10','export')][string]$Profile='research'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (!$ConfigPath) { $ConfigPath = Join-Path $repo 'artifacts.local/work/remote-worker/worker.json' }
if (!(Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw 'Worker config missing. See docs/HOST_RESEARCH_COMPUTE.md and config/remote-worker.example.json.'
}
$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
foreach ($field in @('host','user','identityFile','remotePowerShell','enterScript')) {
    if ([string]::IsNullOrWhiteSpace($config.$field)) { throw "Missing worker field: $field" }
}
if ($config.host -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]*$' -or $config.user -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid SSH endpoint' }
if ($config.remotePowerShell -notmatch '^[A-Za-z]:[\\/][A-Za-z0-9_./\\-]+$') { throw 'remotePowerShell must be a simple absolute executable path without spaces' }
if (!(Test-Path -LiteralPath $config.identityFile -PathType Leaf)) { throw 'SSH identity file missing' }
$enter = $config.enterScript.Replace("'", "''")
$prefix = "`$ErrorActionPreference='Stop'; `$ProgressPreference='SilentlyContinue'; . '$enter' -Profile '$Profile'; if (`$env:BLINDASSIST_WORKER_PROFILE -ne '$Profile') { throw 'Worker profile selection failed' }; "
if ($Action -eq 'Exec') {
    if (!$ScriptFile -or !(Test-Path -LiteralPath $ScriptFile -PathType Leaf)) { throw 'Exec requires a local -ScriptFile' }
    # Windows OpenSSH's command shell has a short command-line limit. Transfer
    # script content with SFTP instead of embedding it in an encoded command.
    $root = Split-Path (Split-Path $config.enterScript -Parent) -Parent
    $remoteScript = ($root.Replace('\','/').TrimEnd('/') + '/artifacts/tmp/invoke-' + [guid]::NewGuid().ToString('N') + '.ps1')
    & scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o IdentitiesOnly=yes -i $config.identityFile (Resolve-Path -LiteralPath $ScriptFile).Path "$($config.user)@$($config.host):$remoteScript"
    if ($LASTEXITCODE -ne 0) { throw "Script upload failed; inspect possible partial upload at $remoteScript" }
    $quoted = $remoteScript.Replace("'", "''")
    $body = "try { $prefix & '$quoted' } finally { Remove-Item -LiteralPath '$quoted'; if (Test-Path -LiteralPath '$quoted') { throw 'Invocation cleanup failed' } }"
    $prefix = ''
} else {
    $body = @'
$source = $env:BLINDASSIST_SOURCE
$head = git -C $source rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Worker source revision unavailable' }
$dirty = git -C $source status --porcelain
$gpu = nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
$gpuExit = $LASTEXITCODE
$os = Get-CimInstance Win32_OperatingSystem
$tasks = @(Get-ScheduledTask -TaskName 'BlindAssist-*' -ErrorAction SilentlyContinue | Select-Object TaskName,State)
[ordered]@{ host=$env:COMPUTERNAME; profile=$env:BLINDASSIST_WORKER_PROFILE; source=$source; revision=$head; dirty=@($dirty); artifacts=$env:BLINDASSIST_ARTIFACTS; python=(Get-Command python).Source; freeMemoryMiB=[math]::Round($os.FreePhysicalMemory/1024); freeDiskBytes=(Get-PSDrive -Name ([IO.Path]::GetPathRoot($source).Substring(0,1))).Free; gpu=$gpu; gpuProbeExit=$gpuExit; jobs=$tasks } | ConvertTo-Json -Depth 5
'@
}
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($prefix + $body))
& ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o IdentitiesOnly=yes -i $config.identityFile "$($config.user)@$($config.host)" "$($config.remotePowerShell) -NoProfile -OutputFormat Text -EncodedCommand $encoded"
if ($LASTEXITCODE -ne 0) { throw "Remote $Action failed with exit code $LASTEXITCODE; do not automatically redispatch an uncertain job." }
