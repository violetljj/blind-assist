[CmdletBinding()]
param(
    [string]$Output = "artifacts.local/evidence/dual-loop/production-temporal-geometry-factorial-ab-r0/input-preflight/input_receipt.json",
    [string]$TruthMembershipOutput = "artifacts.local/evidence/dual-loop/production-temporal-geometry-factorial-ab-r0/input-preflight/truth_membership_receipt.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path "E:\codex-tools\bin" "blindassist-python.cmd"
if (-not (Test-Path -LiteralPath $python)) {
    throw "BlindAssist Python entrypoint not found: $python"
}

$moduleRoot = Join-Path $PSScriptRoot "research/dual_loop_production_temporal_ab"

& $python (
    Join-Path $moduleRoot "input_preflight.py"
) --repo-root $repoRoot --output (Join-Path $repoRoot $Output)
if ($LASTEXITCODE -ne 0) {
    throw "Dual-loop production temporal A/B input preflight failed with exit code $LASTEXITCODE"
}

& $python (
    Join-Path $moduleRoot "truth_membership_preflight.py"
) --repo-root $repoRoot --output (Join-Path $repoRoot $TruthMembershipOutput)
if ($LASTEXITCODE -ne 0) {
    throw "Dual-loop production temporal A/B truth-membership preflight failed with exit code $LASTEXITCODE"
}
