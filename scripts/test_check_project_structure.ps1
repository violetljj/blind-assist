$ErrorActionPreference = 'Stop'
$baseRef = $env:BASE_REF
if (-not $baseRef) {
    $baseRef = (& git rev-parse HEAD~1 2>$null | Select-Object -First 1).Trim()
}
& (Join-Path $PSScriptRoot 'check_project_structure.ps1') -BaseRef $baseRef
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Project structure smoke test passed.'
