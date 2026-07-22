[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$Contract,
    [Parameter(Mandatory = $true)] [string]$Output,
    [ValidateRange(5, 60)] [int]$TimeoutSeconds = 30,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$contractPath = [System.IO.Path]::GetFullPath($Contract)
$outputPath = [System.IO.Path]::GetFullPath($Output)
foreach ($path in @($contractPath, $outputPath)) {
    if ($path.ToLowerInvariant().Replace("_", "-").Contains("secondary-corridor-causal")) {
        throw "Independent-direction paths are forbidden."
    }
}
$spec = Get-Content -Raw -Encoding UTF8 $contractPath | ConvertFrom-Json
$queries = @($spec.queries)
$limit = [int]$spec.request_limits.maximum_results_per_query
$rawDir = "$outputPath.responses"
if ([System.IO.Directory]::Exists($rawDir) -or [System.IO.File]::Exists($outputPath)) {
    throw "Refusing to overwrite Internet Archive discovery evidence."
}
[System.IO.Directory]::CreateDirectory($rawDir) | Out-Null
$headers = @{
    "User-Agent" = "BlindAssist-public-directional-obstruction-audit/1.0"
    "Accept" = "application/json"
}
for ($index = 0; $index -lt $queries.Count; $index++) {
    $tokens = @(([string]$queries[$index]).Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object { "($_)" })
    $archiveQuery = "mediatype:movies AND (" + ($tokens -join " AND ") + ")"
    $fields = @("identifier", "title", "description", "creator", "licenseurl", "subject", "date", "downloads")
    $parts = @("q=$([System.Uri]::EscapeDataString($archiveQuery))")
    $parts += $fields | ForEach-Object { "fl%5B%5D=$([System.Uri]::EscapeDataString($_))" }
    $parts += @("rows=$limit", "page=1", "output=json", "sort%5B%5D=downloads%20desc")
    $url = "https://archive.org/advancedsearch.php?" + ($parts -join "&")
    $destination = Join-Path $rawDir ("{0:D2}.json" -f $index)
    Invoke-WebRequest -UseBasicParsing -Uri $url -Headers $headers -TimeoutSec $TimeoutSeconds -OutFile $destination
    $length = (Get-Item -LiteralPath $destination).Length
    if ($length -le 0 -or $length -gt 5MB) {
        throw "Internet Archive response length is outside the frozen range: $length"
    }
}

$pythonScript = Join-Path $PSScriptRoot "search_internet_archive_public_video_candidates.py"
& $Python $pythonScript --contract $contractPath --responses-dir $rawDir --output $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "Internet Archive candidate parser failed with exit code $LASTEXITCODE."
}
