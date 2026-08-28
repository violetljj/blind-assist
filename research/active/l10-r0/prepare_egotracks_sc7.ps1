[CmdletBinding()]
param(
    [string]$AwsProfileName = 'default',
    [string]$Version = 'v2',
    [int]$CohortLimit = 12
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$python = 'C:\Users\26442\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$ego4dDeps = Join-Path $repoRoot 'artifacts.local\runtime\ego4d-cli-pydeps'
$outputRoot = Join-Path $repoRoot 'artifacts.local\datasets\l10-sc7-egotracks'
$auditRoot = Join-Path $outputRoot 'source-audit'
$credentialPath = Join-Path $env:USERPROFILE '.aws\credentials'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Bundled Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $ego4dDeps)) {
    throw "Ego4D CLI runtime is missing: $ego4dDeps"
}
if (-not (Test-Path -LiteralPath $credentialPath)) {
    throw 'Ego4D AWS credentials have not been provisioned on this machine.'
}

$profilePattern = '^\[' + [regex]::Escape($AwsProfileName) + '\]\s*$'
if (-not (Select-String -LiteralPath $credentialPath -Pattern $profilePattern -Quiet)) {
    throw "AWS profile '$AwsProfileName' is not present. No credential values were read or printed."
}

$env:PYTHONPATH = $ego4dDeps
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $auditRoot -Force | Out-Null

& $python -m ego4d.cli.cli `
    --output_directory $outputRoot `
    --datasets egotracks `
    --version $Version `
    --aws_profile_name $AwsProfileName `
    --no-metadata `
    -y
if ($LASTEXITCODE -ne 0) {
    throw "Official Ego4D CLI failed with exit code $LASTEXITCODE."
}

$annotationFiles = Get-ChildItem -LiteralPath $outputRoot -Recurse -File -Filter '*.json' |
    Where-Object { $_.Name -match '(?i)EgoTracks_(train|val)\.json$' }

foreach ($split in @('train', 'val')) {
    $matches = @($annotationFiles | Where-Object { $_.Name -match "(?i)EgoTracks_$split\.json$" })
    if ($matches.Count -ne 1) {
        throw "Expected exactly one EgoTracks_$split.json annotation; found $($matches.Count)."
    }
    $auditPath = Join-Path $auditRoot "egotracks_sc7_${split}_source_audit.json"
    & $python (Join-Path $PSScriptRoot 'egotracks_sc7_source_audit.py') `
        --annotations $matches[0].FullName `
        --split $split `
        --limit $CohortLimit `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "SC7 source audit failed for split '$split'."
    }
}

[pscustomobject]@{
    Status = 'SC7_EGOTRACKS_SOURCE_AUDIT_COMPLETE'
    DatasetRoot = $outputRoot
    AuditRoot = $auditRoot
    AwsProfile = $AwsProfileName
    SecretsPrinted = $false
} | ConvertTo-Json
