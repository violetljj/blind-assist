$ErrorActionPreference = 'Stop'
$checker = Join-Path $PSScriptRoot 'check_project_structure.ps1'
& $checker
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Project structure smoke test passed.'

# Exercise the actual checker and its exit code in an isolated minimal checkout.
# Never inflate the user's live current documents to test budget enforcement.
$repoRoot = (& git rev-parse --show-toplevel).Trim()
$artifactRoot = Join-Path $repoRoot 'artifacts.local'
$fixture = Join-Path $artifactRoot ('structure-budget-test-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $fixture
$fixture = (Resolve-Path -LiteralPath $fixture).Path
$artifactRoot = (Resolve-Path -LiteralPath $artifactRoot).Path
$pushed = $false
try {
    foreach ($relative in @(
        'AGENTS.md', 'docs/PROJECT_STATE.md', 'docs/CURRENT_DECISION.md',
        'docs/history-index.md', 'research/active/dtr-r0/README.md',
        'research/active/dtr-r0/CURRENT.md', 'research/active/l10-r0/README.md',
        'research/active/l10-r0/CURRENT.md', 'research/active/hardware-bringup/README.md',
        'research/active/hardware-bringup/CURRENT.md',
        'research/active/l10-r0/l10_r0.py', 'research/active/l10-r0/benchmark.py',
        'research/active/l10-r0/artvideo_replay.py', 'research/active/dtr-r0/pyproject.toml',
        'research/active/dtr-r0/dtr_r0.py', 'research/active/dtr-r0/real_observation_adapter.py',
        'research/active/dtr-r0/test_real_observation_adapter.py', 'tools/ba.ps1',
        'scripts/show_worktree_scope.ps1', 'scripts/refresh_knowledge.ps1',
        '.githooks/pre-commit', 'config/local.example.toml',
        '.codex/environments/environment.toml', '.worktreeinclude',
        'experiments/index.jsonl', 'data/dataset-ledger-summary.csv',
        'data/dataset-ledger-manifest.json'
    )) {
        $path = Join-Path $fixture $relative
        $null = New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force
        [IO.File]::WriteAllText($path, '')
    }
    [IO.File]::WriteAllText((Join-Path $fixture '.worktreeinclude'), "config/local.toml`nlocal.properties`n")
    & git -c core.hideDotFiles=false init --quiet --template= $fixture
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize structure fixture.' }
    # Only route names are read from the index; no commits or user identity needed.
    foreach ($route in @('dtr-r0', 'hardware-bringup', 'l10-r0')) {
        & git -C $fixture update-index --add --cacheinfo "100644,e69de29bb2d1d6434b8b29ae775ad8c2e48c5391,research/active/$route/CURRENT.md"
        if ($LASTEXITCODE -ne 0) { throw 'Could not index structure fixture routes.' }
    }
    Push-Location $fixture
    $pushed = $true
    $current = Join-Path $fixture 'research/active/dtr-r0/CURRENT.md'
    foreach ($case in @(
        @{ Name = 'exact line and byte boundary'; Text = ('x' * (16384 - 150)) + ("`n" * 150); Exit = 0 },
        @{ Name = 'byte overflow only'; Text = ('x' * 16385); Exit = 1 },
        @{ Name = 'line overflow only'; Text = ("x`n" * 151); Exit = 1 },
        @{ Name = 'UTF-8 byte overflow below character limit'; Text = ([string][char]0x4e2d * 5462); Exit = 1 }
    )) {
        [IO.File]::WriteAllText($current, $case.Text, [Text.UTF8Encoding]::new($false))
        $output = @(& pwsh -NoProfile -File $checker 2>&1)
        $code = $LASTEXITCODE
        if ($code -ne $case.Exit) {
            throw "$($case.Name): expected exit $($case.Exit), got $code.`n$($output -join "`n")"
        }
        if ($case.Exit -eq 1 -and ($output -join "`n") -notmatch 'Compact current exceeds budget: research/active/dtr-r0/CURRENT.md') {
            throw "$($case.Name): failed for an unrelated reason.`n$($output -join "`n")"
        }
        Write-Host "Document budget regression passed: $($case.Name)."
    }
}
finally {
    if ($pushed) { Pop-Location }
    # Only this test's direct child tree may be removed, with no nested junctions.
    if ((Split-Path -Parent $fixture) -ne $artifactRoot) { throw "Unsafe fixture cleanup path: $fixture" }
    $links = @(Get-Item -LiteralPath $fixture) + @(Get-ChildItem -LiteralPath $fixture -Recurse -Force)
    if ($links | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) {
        throw "Unexpected reparse point in fixture; retained for inspection: $fixture"
    }
    Remove-Item -LiteralPath $fixture -Recurse
    if (Test-Path -LiteralPath $fixture) { throw "Fixture cleanup failed: $fixture" }
}
Write-Host 'Project structure and document budget tests passed.'
exit 0
