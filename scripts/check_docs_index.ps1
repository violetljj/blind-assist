$ErrorActionPreference = 'Stop'
$repoRoot = (& git rev-parse --show-toplevel 2>$null | Select-Object -First 1).Trim()
if (-not $repoRoot) { throw 'Run inside the BlindAssist Git checkout.' }
$sources = @(
    'README.md', 'docs/README.md', 'docs/PROJECT_STATE.md',
    'docs/CURRENT_DECISION.md', 'docs/DOCUMENT_GOVERNANCE.md',
    'docs/history-index.md', 'scripts/README.md',
    'research/active/l10-r0/CURRENT.md',
    'research/active/dtr-r0/CURRENT.md'
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

function Require-Literal([string]$Relative, [string]$Needle, [string]$Description) {
    $path = Join-Path $repoRoot $Relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return }
    $text = Get-Content -LiteralPath $path -Raw -Encoding utf8
    if (-not $text.Contains($Needle)) {
        $script:failures.Add("Semantic drift in $Relative`: missing $Description")
    }
}

Require-Literal 'README.md' 'research/active/l10-r0/CURRENT.md' 'L10 compact-current link'
Require-Literal 'README.md' 'research/active/dtr-r0/CURRENT.md' 'DTR compact-current link'
Require-Literal 'docs/README.md' 'Current Dynamic Travel Risk R2 route' 'DTR R2 route label'
Require-Literal 'docs/PROJECT_STATE.md' 'L10_R0_ACTIVE' 'L10 active status'
Require-Literal 'docs/PROJECT_STATE.md' 'DTR_R2_DYNAMIC_RETAINED' 'DTR R2 status'
Require-Literal 'docs/CURRENT_DECISION.md' 'L10_R0_ACTIVE / DTR_R2_DYNAMIC_RETAINED' 'cross-route status'
Require-Literal 'research/active/l10-r0/CURRENT.md' 'Status: `L10_R0_ACTIVE`' 'L10 route status'
Require-Literal 'research/active/dtr-r0/CURRENT.md' 'Status: `DTR_R2_DYNAMIC_RETAINED`' 'DTR route status'

if ($failures.Count) {
    Write-Host 'Documentation index check failed:'
    $failures | ForEach-Object { Write-Host " - $_" }
    exit 1
}
Write-Host "Documentation index check passed for $($sources.Count) hot files and $checked local links."
