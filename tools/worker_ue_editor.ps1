# Run interactively on the worker desktop (for example through UU Remote).
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$WorkerRoot,
    [switch]$ShowCommand
)
$ErrorActionPreference='Stop'
$settings=Get-Content -LiteralPath (Join-Path $WorkerRoot 'config/environment.json') -Raw | ConvertFrom-Json
if (!$settings.ueDevelopmentProject -or !$settings.ueEngineRoot) { throw 'UE development paths are not configured' }
$project=[IO.Path]::GetFullPath($settings.ueDevelopmentProject)
$editor=Join-Path $settings.ueEngineRoot 'Engine/Binaries/Win64/UnrealEditor.exe'
if (!(Test-Path -LiteralPath $project -PathType Leaf) -or !(Test-Path -LiteralPath $editor -PathType Leaf)) { throw 'Configured UE editor/project is missing' }
if ($project.Contains('"') -or $editor.Contains('"')) { throw 'Invalid quote in configured path' }
$argsList=@(('"'+$project+'"'), '-NoSplash')
if ($ShowCommand) {
    [ordered]@{editor=$editor;project=$project;arguments=$argsList;ddc=$settings.ueDdcPath;desktop='UU Remote'} | ConvertTo-Json
    return
}
if ($settings.ueDdcPath) { [Environment]::SetEnvironmentVariable('UE-LocalDataCachePath', $settings.ueDdcPath, 'Process') }
# The user starts this visible editor from their interactive desktop session.
Start-Process -FilePath $editor -ArgumentList $argsList -WorkingDirectory (Split-Path $project) -WindowStyle Normal
