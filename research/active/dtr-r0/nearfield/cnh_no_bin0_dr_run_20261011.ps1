param(
    [Parameter(Mandatory=$true)][string]$Script,
    [Parameter(Mandatory=$true)][string]$Stage,
    [ValidateSet('GPU','CPU')][string]$Backend = 'GPU',
    [string[]]$Arguments = @()
)
$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\linnan\linnan'
$taskOutput = Join-Path $taskRoot 'artifacts.local\work\cnh-no-bin0-dr-retrain-dev-20261011'
$receiptFolder = Join-Path $taskOutput 'command_wall_receipts'
New-Item -ItemType Directory -Path $receiptFolder -ErrorAction SilentlyContinue | Out-Null
$resolvedScript = (Resolve-Path -LiteralPath (Join-Path $taskRoot $Script)).Path
if (-not $resolvedScript.StartsWith($taskRoot + '\',[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Script outside task checkout' }
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$code = -1
try {
    foreach ($cacheName in @('TEMP','TMP','CUPY_CACHE_DIR','NUMBA_CACHE_DIR','TORCH_HOME')) {
        $cachePath = Join-Path $taskOutput ('runtime\' + $cacheName.ToLowerInvariant())
        New-Item -ItemType Directory -Path $cachePath -ErrorAction SilentlyContinue | Out-Null
        [Environment]::SetEnvironmentVariable($cacheName,$cachePath,'Process')
    }
    Push-Location $taskRoot
    try {
        if ($Backend -eq 'GPU') { & 'E:\codex-tools\bin\blindassist-research-gpu.cmd' $resolvedScript @Arguments }
        else { & 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' $resolvedScript @Arguments }
        $code = $LASTEXITCODE
    } finally { Pop-Location }
} finally {
    $watch.Stop()
    $receipt = @{ stage=$Stage; backend=$Backend; seconds=$watch.Elapsed.TotalSeconds; exit_code=$code; script=$Script; arguments=$Arguments; utc=[DateTime]::UtcNow.ToString('o') }
    $receiptName = $Stage + '_' + [DateTime]::UtcNow.Ticks + '.json'
    $receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $receiptFolder $receiptName) -Encoding utf8
}
if ($code -ne 0) { throw "Task stage $Stage failed (exit $code); command-wall receipt preserved" }
