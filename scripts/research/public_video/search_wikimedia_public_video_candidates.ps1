[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$Contract,
    [Parameter(Mandatory = $true)] [string]$Output,
    [ValidateRange(5, 60)] [int]$TimeoutSeconds = 30,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$forbidden = "secondary-corridor-causal"
$outputPath = [System.IO.Path]::GetFullPath($Output)
if ($outputPath.ToLowerInvariant().Replace("_", "-").Contains($forbidden)) {
    throw "Independent-direction output paths are forbidden."
}
$contractPath = [System.IO.Path]::GetFullPath($Contract)
$spec = Get-Content -Raw -Encoding UTF8 $contractPath | ConvertFrom-Json
$queries = @($spec.queries.wikimedia_commons)
$limit = [int]$spec.request_limits.maximum_results_per_query
$rawDir = "$outputPath.responses"
if ([System.IO.Directory]::Exists($rawDir) -or [System.IO.File]::Exists($outputPath)) {
    throw "Refusing to overwrite Commons discovery evidence."
}
[System.IO.Directory]::CreateDirectory($rawDir) | Out-Null
$headers = @{
    "User-Agent" = "BlindAssist-public-directional-obstruction-audit/1.0"
    "Accept" = "application/json"
}
for ($index = 0; $index -lt $queries.Count; $index++) {
    $query = [System.Uri]::EscapeDataString([string]$queries[$index])
    $url = "https://commons.wikimedia.org/w/api.php?action=query&format=json&formatversion=2&generator=search&gsrsearch=$query&gsrnamespace=6&gsrlimit=$limit&prop=imageinfo&iiprop=url%7Cmime%7Csize%7Cextmetadata"
    $destination = Join-Path $rawDir ("{0:D2}.json" -f $index)
    Invoke-WebRequest -UseBasicParsing -Uri $url -Headers $headers -TimeoutSec $TimeoutSeconds -OutFile $destination
    $length = (Get-Item -LiteralPath $destination).Length
    if ($length -le 0 -or $length -gt 5MB) {
        throw "Commons response length is outside the frozen range: $length"
    }
}

$pythonScript = Join-Path $PSScriptRoot "search_wikimedia_public_video_candidates.py"
& $Python $pythonScript --contract $contractPath --responses-dir $rawDir --output $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "Commons candidate parser failed with exit code $LASTEXITCODE."
}
