param(
    [string]$Plan = 'experiments/city-field/native-field-v2.json',
    [string]$Output,
    [string]$MachineConfig = 'artifacts.local/city-field-machine.json',
    [switch]$CompileOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
if (-not $Output) { $Output = 'artifacts.local/nearfield/city-field-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
if (Test-Path -LiteralPath $MachineConfig) {
    $machine = Get-Content -LiteralPath $MachineConfig -Raw | ConvertFrom-Json
} else {
    throw "Missing local machine paths: $MachineConfig. See experiments/city-field/README.md."
}
foreach ($name in @('python','project','engine','plugin')) {
    if (-not (Test-Path -LiteralPath $machine.$name)) { throw "Missing $name path in $MachineConfig" }
}
$arguments = @('tools/run_city_field_collection.py','--plan',$Plan,'--output',$Output,
    '--project',$machine.project,'--engine',$machine.engine,'--plugin',$machine.plugin)
if ($CompileOnly) { $arguments += '--compile-only' }
& $machine.python @arguments
if ($LASTEXITCODE -ne 0) { throw "City collection failed; preserved receipts: $Output" }
Write-Host "City collection output: $Output"
if (-not $CompileOnly) { Write-Host "Usable data: $Output/ready/frames.jsonl ; excluded frames: $Output/ready/excluded.json" }
