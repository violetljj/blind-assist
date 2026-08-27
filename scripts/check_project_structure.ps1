param([string]$BaseRef = $env:BASE_REF)

$ErrorActionPreference = 'Stop'
$repoRoot = (& git rev-parse --show-toplevel 2>$null | Select-Object -First 1).Trim()
if (-not $repoRoot) { throw 'Run inside the BlindAssist Git checkout.' }
$failures = [Collections.Generic.List[string]]::new()

function Require-Path([string]$Relative, [string]$Type = 'Leaf') {
    $path = Join-Path $repoRoot $Relative
    if (-not (Test-Path -LiteralPath $path -PathType $Type)) {
        $script:failures.Add("Missing required $Type`: $Relative")
    }
}

foreach ($file in @(
    'AGENTS.md', 'docs/PROJECT_STATE.md', 'docs/CURRENT_DECISION.md',
    'docs/history-index.md', 'research/active/dtr-r0/README.md',
    'research/active/dtr-r0/pyproject.toml',
    'research/active/dtr-r0/dtr_r0.py',
    'research/active/dtr-r0/real_observation_adapter.py',
    'research/active/dtr-r0/test_real_observation_adapter.py', 'tools/ba.ps1',
    'config/local.example.toml', '.codex/environments/environment.toml',
    '.worktreeinclude', 'experiments/index.jsonl',
    'data/dataset-ledger-summary.csv', 'data/dataset-ledger-manifest.json'
)) { Require-Path $file }

$trackedActiveFiles = @(& git -C $repoRoot ls-files -- 'research/active/*')
$active = @($trackedActiveFiles | ForEach-Object {
    if ($_ -match '^research/active/([^/]+)/') { $Matches[1] }
} | Sort-Object -Unique)
if ($active.Count -ne 1) { $failures.Add("Expected exactly one tracked active research route; found $($active.Count).") }

$agents = Join-Path $repoRoot 'AGENTS.md'
if (Test-Path $agents) {
    $agentLines = [IO.File]::ReadAllLines($agents).Count
    $agentBytes = (Get-Item $agents).Length
    if ($agentLines -gt 150 -or $agentBytes -gt 10240) {
        $failures.Add("AGENTS.md exceeds 150 lines or 10 KiB ($agentLines lines, $agentBytes bytes).")
    }
}

$log = Join-Path $repoRoot 'DEVELOPMENT_LOG.md'
if (Test-Path $log) {
    $logLines = [IO.File]::ReadAllLines($log).Count
    $logBytes = (Get-Item $log).Length
    if ($logLines -gt 200 -or $logBytes -gt 102400) {
        $failures.Add("DEVELOPMENT_LOG.md exceeds 200 lines or 100 KiB ($logLines lines, $logBytes bytes).")
    }
}

foreach ($forbidden in @(
    'DATASET_MASTER_LEDGER.json', 'DATASET_MASTER_LEDGER.csv',
    'scripts/research', 'docs/research', 'docs/history', 'schemas'
)) {
    if (Test-Path (Join-Path $repoRoot $forbidden)) { $failures.Add("Cold surface remains in current tree: $forbidden") }
}

$topScripts = @(Get-ChildItem (Join-Path $repoRoot 'scripts') -File)
if ($topScripts.Count -gt 25) { $failures.Add("Top-level scripts exceed 25 files: $($topScripts.Count)") }
$topDocs = @(Get-ChildItem (Join-Path $repoRoot 'docs') -File)
if ($topDocs.Count -gt 25) { $failures.Add("Top-level docs exceed 25 files: $($topDocs.Count)") }

$hotFiles = @(
    'AGENTS.md', 'README.md', 'docs/PROJECT_STATE.md',
    'docs/CURRENT_DECISION.md', 'research/active/dtr-r0/README.md',
    'config/local.example.toml'
)
foreach ($relative in $hotFiles) {
    $path = Join-Path $repoRoot $relative
    if (-not (Test-Path $path)) { continue }
    $text = Get-Content $path -Raw -Encoding utf8
    if ($text -match '(?i)(?:[A-Z]:[\\/](?:Users|linnan|codex-tools)|/home/[^/]+/)') {
        $failures.Add("Machine-specific absolute path in hot file: $relative")
    }
}

$ledger = Join-Path $repoRoot 'experiments/index.jsonl'
if (Test-Path $ledger) {
    $lineNumber = 0
    foreach ($line in Get-Content $ledger -Encoding utf8) {
        $lineNumber++
        if (-not $line.Trim()) { continue }
        try { $null = $line | ConvertFrom-Json -Depth 20 }
        catch { $failures.Add("Malformed experiments/index.jsonl line $lineNumber`: $($_.Exception.Message)") }
    }
}

$include = Join-Path $repoRoot '.worktreeinclude'
if (Test-Path $include) {
    $includeText = Get-Content $include -Raw
    foreach ($entry in @('config/local.toml', 'local.properties')) {
        if ($includeText -notmatch [regex]::Escape($entry)) { $failures.Add(".worktreeinclude lacks $entry") }
    }
}

if (Test-Path (Join-Path $repoRoot 'config/local.toml')) {
    & git -C $repoRoot check-ignore -q -- config/local.toml
    if ($LASTEXITCODE -ne 0) { $failures.Add('config/local.toml must be ignored.') }
}

if ($failures.Count) {
    Write-Host 'Project structure check failed:'
    $failures | ForEach-Object { Write-Host " - $_" }
    exit 1
}
Write-Host "Project structure check passed: tracked_active=$($active[0]), scripts=$($topScripts.Count), docs=$($topDocs.Count), AGENTS=$agentLines lines/$agentBytes bytes."
