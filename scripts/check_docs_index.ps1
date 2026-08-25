$ErrorActionPreference = 'Stop'
$repoRoot = (& git rev-parse --show-toplevel 2>$null | Select-Object -First 1).Trim()
if (-not $repoRoot) { throw 'Run inside the BlindAssist Git checkout.' }
$sources = @(
    'README.md', 'docs/README.md', 'docs/PROJECT_STATE.md',
    'docs/CURRENT_DECISION.md', 'docs/DOCUMENT_GOVERNANCE.md',
    'docs/history-index.md', 'scripts/README.md',
    'research/active/grail-r1cl/README.md'
)
$failures = [Collections.Generic.List[string]]::new()
$checked = 0
$pattern = '\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)'

foreach ($relative in $sources) {
    $source = Join-Path $repoRoot $relative
    if (-not (Test-Path $source -PathType Leaf)) {
        $failures.Add("Missing hot documentation source: $relative")
        continue
    }
    $text = Get-Content $source -Raw -Encoding utf8
    foreach ($match in [regex]::Matches($text, $pattern)) {
        $target = $match.Groups[1].Value.Trim().Trim('<','>')
        if ($target -match '^[A-Za-z][A-Za-z0-9+.-]*:' -or $target.StartsWith('/')) { continue }
        $checked++
        $candidate = [IO.Path]::GetFullPath((Join-Path (Split-Path $source -Parent) $target))
        if (-not (Test-Path -LiteralPath $candidate)) {
            $failures.Add("Missing link target in $relative`: $target")
        }
    }
}

if ($failures.Count) {
    Write-Host 'Documentation index check failed:'
    $failures | ForEach-Object { Write-Host " - $_" }
    exit 1
}
Write-Host "Documentation index check passed for $($sources.Count) hot files and $checked local links."
