param([Parameter(Mandatory=$true)][string]$Script,[Parameter(Mandatory=$true)][string]$Stage,
      [ValidateSet('GPU','CPU')][string]$Backend='CPU',[string[]]$Arguments=@())
$ErrorActionPreference='Stop'
$taskRoot='E:\linnan\linnan'
$taskOut=Join-Path $taskRoot 'artifacts.local\work\local-signed-bias-contract-dev-20261011'
$receiptFolder=Join-Path $taskOut 'command_wall_receipts'
New-Item -ItemType Directory -Path $receiptFolder -ErrorAction SilentlyContinue | Out-Null
$taskScript=(Resolve-Path -LiteralPath (Join-Path $taskRoot $Script)).Path
if (-not $taskScript.StartsWith($taskRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Script outside checkout' }
$taskWatch=[Diagnostics.Stopwatch]::StartNew();$taskCode=-1
try {
    foreach($cacheName in @('TEMP','TMP','CUPY_CACHE_DIR','NUMBA_CACHE_DIR','TORCH_HOME','PYTHONPYCACHEPREFIX')) {
        $cachePath=Join-Path $taskOut ('runtime\'+$cacheName.ToLowerInvariant())
        New-Item -ItemType Directory -Path $cachePath -ErrorAction SilentlyContinue | Out-Null
        [Environment]::SetEnvironmentVariable($cacheName,$cachePath,'Process')
    }
    Push-Location $taskRoot
    try {
        if($Backend -eq 'GPU'){ & 'E:\codex-tools\bin\blindassist-research-gpu.cmd' $taskScript @Arguments }
        else { & 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' $taskScript @Arguments }
        $taskCode=$LASTEXITCODE
    } finally { Pop-Location }
} finally {
    $taskWatch.Stop()
    @{stage=$Stage;backend=$Backend;seconds=$taskWatch.Elapsed.TotalSeconds;exit_code=$taskCode;script=$Script;arguments=$Arguments;utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $receiptFolder ($Stage+'_'+[DateTime]::UtcNow.Ticks+'.json')) -Encoding utf8
}
if($taskCode -ne 0){throw "Stage $Stage failed ($taskCode), receipt preserved"}
