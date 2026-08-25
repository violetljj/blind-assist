$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'check_project_structure.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Project structure smoke test passed.'
