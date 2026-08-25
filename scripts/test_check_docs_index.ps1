$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'check_docs_index.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Documentation index smoke test passed.'
