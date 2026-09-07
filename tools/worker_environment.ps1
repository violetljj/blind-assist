# Dot-source on a provisioned worker. Machine paths remain in worker-local JSON.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$WorkerRoot,
    [ValidateSet('research','l10','export')][string]$Profile='research'
)
$ErrorActionPreference='Stop'
$settings=Get-Content -LiteralPath (Join-Path $WorkerRoot 'config/environment.json') -Raw | ConvertFrom-Json
$source=Join-Path $WorkerRoot ('source/'+$settings.revision)
$data=Join-Path $WorkerRoot 'artifacts'
$venv=Join-Path $data ('work/envs/'+$Profile)
if (!(Test-Path -LiteralPath "$venv/Scripts/python.exe")) { throw "Worker profile is not provisioned: $Profile" }
if (!(Test-Path -LiteralPath "$source/.git")) { throw 'Worker source is missing' }
if (!$env:BLINDASSIST_BASE_PATH) { $env:BLINDASSIST_BASE_PATH=$env:PATH }
$env:BLINDASSIST_WORKER_ROOT=$WorkerRoot
$env:BLINDASSIST_SOURCE=$source
$env:BLINDASSIST_ARTIFACTS=$data
$env:BLINDASSIST_WORKER_PROFILE=$Profile
if ($settings.ueEngineRoot) { $env:UE_ENGINE_ROOT=$settings.ueEngineRoot }
if ($settings.ueCapturePlugin) { $env:BLINDASSIST_UE_CAPTURE_PLUGIN=$settings.ueCapturePlugin }
if ($settings.ueDevelopmentProject) { $env:BLINDASSIST_UE_DEVELOPMENT_PROJECT=$settings.ueDevelopmentProject }
if ($settings.ueDdcPath) { [Environment]::SetEnvironmentVariable('UE-LocalDataCachePath', $settings.ueDdcPath, 'Process') }
if ($settings.msvcRoot) { $env:BLINDASSIST_MSVC_ROOT=$settings.msvcRoot }
$env:VIRTUAL_ENV=$venv
$env:PYTHONUTF8='1'
$env:CUDA_PATH=Join-Path $data 'work/toolchains/cuda128'
$env:BLINDASSIST_JAVA_HOME=$settings.javaHome
$env:JAVA_HOME=$settings.javaHome
$env:BLINDASSIST_ANDROID_SDK_ROOT=Join-Path $data 'work/toolchains/android-sdk'
$env:ANDROID_HOME=$env:BLINDASSIST_ANDROID_SDK_ROOT
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
$env:BLINDASSIST_ANDROID_USER_HOME=Join-Path $data 'work/cache/android-user'
$env:ANDROID_USER_HOME=$env:BLINDASSIST_ANDROID_USER_HOME
$env:BLINDASSIST_GRADLE_USER_HOME=Join-Path $data 'work/cache/gradle'
$env:GRADLE_USER_HOME=$env:BLINDASSIST_GRADLE_USER_HOME
$env:PIP_CACHE_DIR=Join-Path $data 'work/cache/pip'
$env:HF_HOME=Join-Path $data 'models/huggingface'
$env:TORCH_HOME=Join-Path $data 'models/torch'
$env:XDG_CACHE_HOME=Join-Path $data 'work/cache/xdg'
$env:MPLCONFIGDIR=Join-Path $data 'work/cache/matplotlib'
$env:CUPY_CACHE_DIR=Join-Path $data 'work/cache/cupy'
$env:NUMBA_CACHE_DIR=Join-Path $data 'work/cache/numba'
$env:YOLO_CONFIG_DIR=Join-Path $data 'work/cache/ultralytics'
$env:npm_config_cache=Join-Path $data 'work/cache/npm'
$env:BLINDASSIST_RESEARCH_PYTHON=Join-Path $data 'work/envs/research/Scripts/python.exe'
$env:TEMP=Join-Path $data 'tmp'
$env:TMP=$env:TEMP
foreach($folder in @($env:TEMP,$env:MPLCONFIGDIR,$env:CUPY_CACHE_DIR,$env:NUMBA_CACHE_DIR,$env:YOLO_CONFIG_DIR,$env:npm_config_cache,$env:ANDROID_USER_HOME)) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
}
$sharedTools=Join-Path (Split-Path $WorkerRoot -Parent) 'tools'
$env:PATH=@("$venv/Scripts",$sharedTools,"$sharedTools/pwsh","$env:CUDA_PATH/bin","$env:CUDA_PATH/nvvm/bin","$venv/Lib/site-packages/torch/lib","$env:JAVA_HOME/bin","$env:ANDROID_HOME/platform-tools","$env:ANDROID_HOME/cmake/3.22.1/bin",$env:BLINDASSIST_BASE_PATH) -join ';'
Set-Location -LiteralPath $source
