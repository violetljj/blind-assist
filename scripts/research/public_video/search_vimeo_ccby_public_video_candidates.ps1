[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Query,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [ValidateRange(1, 50)]
    [int]$MaxResults = 20,

    [ValidateRange(5, 60)]
    [int]$TimeoutSeconds = 20,

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$forbiddenToken = "secondary-corridor-causal"
$outputPath = [System.IO.Path]::GetFullPath($Output)
if ($outputPath.ToLowerInvariant().Replace("_", "-").Contains($forbiddenToken)) {
    throw "Independent-direction output paths are forbidden."
}

$outputDirectory = [System.IO.Path]::GetDirectoryName($outputPath)
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
}
$htmlPath = "$outputPath.source.html"
if ([System.IO.File]::Exists($htmlPath)) {
    throw "Raw search evidence already exists: $htmlPath"
}

$encodedQuery = [System.Uri]::EscapeDataString($Query.Trim())
if ([string]::IsNullOrWhiteSpace($encodedQuery)) {
    throw "Search query must not be empty."
}
$searchUrl = "https://vimeo.com/creativecommons/by?search=$encodedQuery"
$headers = @{
    "User-Agent" = "BlindAssist-public-candidate-audit/1.0"
    "Accept" = "text/html,application/xhtml+xml"
    "Accept-Language" = "en-US,en;q=0.8"
}

# Exactly one network request. No pagination, retry loop, cookies, or login.
$response = Invoke-WebRequest `
    -UseBasicParsing `
    -Uri $searchUrl `
    -Headers $headers `
    -TimeoutSec $TimeoutSeconds `
    -OutFile $htmlPath `
    -PassThru

if ($response.StatusCode -ne 200) {
    throw "Vimeo returned HTTP $($response.StatusCode)."
}
$htmlLength = (Get-Item -LiteralPath $htmlPath).Length
if ($htmlLength -le 0 -or $htmlLength -gt 5MB) {
    throw "Vimeo response length is outside the frozen range: $htmlLength bytes."
}

$pythonScript = Join-Path $PSScriptRoot "search_vimeo_ccby_public_video_candidates.py"
& $Python $pythonScript `
    --query $Query `
    --html $htmlPath `
    --html-retrieval-mode online_single_page_external_fetch `
    --max-results $MaxResults `
    --output $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "Candidate-ledger parser failed with exit code $LASTEXITCODE."
}

[pscustomobject]@{
    SearchUrl = $searchUrl
    RawHtmlPath = $htmlPath
    RawHtmlBytes = $htmlLength
    LedgerPath = $outputPath
}
