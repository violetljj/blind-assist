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
    'research/active/dtr-r0/CURRENT.md',
    'research/active/l10-r0/README.md',
    'research/active/l10-r0/CURRENT.md',
    'research/active/l10-r0/l10_r0.py',
    'research/active/l10-r0/benchmark.py',
    'research/active/l10-r0/artvideo_replay.py',
    'research/active/dtr-r0/pyproject.toml',
    'research/active/dtr-r0/dtr_r0.py',
    'research/active/dtr-r0/real_observation_adapter.py',
    'research/active/dtr-r0/test_real_observation_adapter.py', 'tools/ba.ps1',
    'scripts/show_worktree_scope.ps1',
    'config/local.example.toml', '.codex/environments/environment.toml',
    '.worktreeinclude', 'experiments/index.jsonl',
    'data/dataset-ledger-summary.csv', 'data/dataset-ledger-manifest.json'
)) { Require-Path $file }

$trackedActiveFiles = @(& git -C $repoRoot ls-files -- 'research/active/*')
$active = @($trackedActiveFiles | ForEach-Object {
    if ($_ -match '^research/active/([^/]+)/') { $Matches[1] }
} | Sort-Object -Unique)
$expectedActive = @('dtr-r0', 'l10-r0')
if (($active -join ',') -ne ($expectedActive -join ',')) {
    $failures.Add("Expected active routes $($expectedActive -join ','); found $($active -join ',').")
}

$agents = Join-Path $repoRoot 'AGENTS.md'
if (Test-Path $agents) {
    $agentLines = [IO.File]::ReadAllLines($agents).Count
    $agentBytes = (Get-Item $agents).Length
    if ($agentLines -gt 120 -or $agentBytes -gt 8192) {
        $failures.Add("AGENTS.md exceeds 120 lines or 8 KiB ($agentLines lines, $agentBytes bytes).")
    }
}

$compactBudgets = @(
    [pscustomobject]@{ Path = 'docs/PROJECT_STATE.md'; Lines = 200; Bytes = 20480 },
    [pscustomobject]@{ Path = 'docs/CURRENT_DECISION.md'; Lines = 200; Bytes = 20480 },
    [pscustomobject]@{ Path = 'research/active/l10-r0/CURRENT.md'; Lines = 150; Bytes = 16384 },
    [pscustomobject]@{ Path = 'research/active/dtr-r0/CURRENT.md'; Lines = 150; Bytes = 16384 }
)
foreach ($budget in $compactBudgets) {
    $path = Join-Path $repoRoot $budget.Path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
    $lineCount = [IO.File]::ReadAllLines($path).Count
    $byteCount = (Get-Item -LiteralPath $path).Length
    if ($lineCount -gt $budget.Lines -or $byteCount -gt $budget.Bytes) {
        $failures.Add("Compact current exceeds budget: $($budget.Path) ($lineCount lines, $byteCount bytes).")
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
    'docs/CURRENT_DECISION.md', 'research/active/dtr-r0/CURRENT.md',
    'research/active/l10-r0/CURRENT.md',
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
Write-Host "Project structure check passed: tracked_active=$($active -join ','), scripts=$($topScripts.Count), docs=$($topDocs.Count), AGENTS=$agentLines lines/$agentBytes bytes."
