param(
    [Parameter(Mandatory = $true)][string]$DatasetRoot,
    [Parameter(Mandatory = $true)][string]$LinkManifest,
    [Parameter(Mandatory = $true)][string]$ImagePaths,
    [switch]$SkipFinalCompleteness
)

$ErrorActionPreference = 'Stop'
$dataset = [System.IO.Path]::GetFullPath($DatasetRoot)
$links = Get-Content -Raw -LiteralPath $LinkManifest | ConvertFrom-Json
$wanted = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
foreach ($path in Get-Content -LiteralPath $ImagePaths) {
    if ($path) { [void]$wanted.Add($path.Replace('\', '/')) }
}
$receipts = Join-Path $dataset 'download_receipts'
New-Item -ItemType Directory -Force -Path $receipts | Out-Null

foreach ($item in $links) {
    $archive = Join-Path $dataset $item.name
    $receipt = Join-Path $receipts ($item.name + '.json')
    if (Test-Path -LiteralPath $receipt) {
        $prior = Get-Content -Raw -LiteralPath $receipt | ConvertFrom-Json
        if ($prior.status -eq 'COMPLETE') {
            Write-Output "skip complete $($item.name)"
            continue
        }
    }

    Write-Output "download $($item.name)"
    & curl.exe --silent --show-error -L --fail --retry 5 --retry-delay 5 `
        --connect-timeout 30 --speed-limit 1024 --speed-time 60 `
        --continue-at - --output $archive $item.url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $($item.name)" }
    $actual = (Get-FileHash -Algorithm MD5 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($actual -ne $item.md5) { throw "MD5 mismatch for $($item.name): $actual" }

    $volumeList = Join-Path $receipts ($item.name + '.paths.txt')
    $selected = @(& tar -tf $archive | Where-Object { $wanted.Contains($_) })
    if ($LASTEXITCODE -ne 0) { throw "archive listing failed for $($item.name)" }
    $selected | Set-Content -LiteralPath $volumeList -Encoding utf8
    if ($selected.Count -gt 0) {
        & tar -xf $archive -C $dataset -T $volumeList
        if ($LASTEXITCODE -ne 0) { throw "selective extraction failed for $($item.name)" }
        $missing = @($selected | Where-Object { -not (Test-Path -LiteralPath (Join-Path $dataset $_)) })
        if ($missing.Count -gt 0) { throw "$($missing.Count) selected files missing after $($item.name)" }
    }
    [ordered]@{
        status = 'COMPLETE'
        archive = $item.name
        md5 = $actual
        selected_image_count = $selected.Count
        archive_deleted_after_verified_extraction = $true
    } | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding utf8
    Remove-Item -LiteralPath $archive
    Write-Output "complete $($item.name): $($selected.Count) selected images"
}

if (-not $SkipFinalCompleteness) {
    $missingAll = @(Get-Content -LiteralPath $ImagePaths | Where-Object {
        $_ -and -not (Test-Path -LiteralPath (Join-Path $dataset $_))
    })
    if ($missingAll.Count -gt 0) { throw "$($missingAll.Count) selected MSLS images remain missing" }
    Write-Output "all selected MSLS images present: $($wanted.Count)"
}
