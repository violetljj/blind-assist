param(
    [Parameter(Mandatory=$true)][string]$Verification,
    [Parameter(Mandatory=$true)][string]$Receipt
)
$ErrorActionPreference='Stop'
$check=Get-Content -LiteralPath $Verification -Raw | ConvertFrom-Json -AsHashtable
if($check.status -ne 'PASS_CANONICAL_ENCODING'){throw 'Canonical verification must pass first'}
$captureRoot=[IO.Path]::GetFullPath($check.capture).TrimEnd('\')
$repoRoot=Split-Path $PSScriptRoot -Parent
$artifactEntry=Get-Item -LiteralPath (Join-Path $repoRoot 'artifacts.local')
$artifactRoot=if($artifactEntry.LinkType){$artifactEntry.ResolveLinkTarget($true).FullName}else{$artifactEntry.FullName}
if(-not $captureRoot.StartsWith($artifactRoot.TrimEnd('\')+'\evidence\',[StringComparison]::OrdinalIgnoreCase)){throw 'Capture must remain inside canonical evidence storage'}
$spec=Get-Content -LiteralPath (Join-Path $captureRoot 'source/spec.json') -Raw | ConvertFrom-Json
if($spec.scope -ne 'STREET_DEVELOPMENT_PILOT_NOT_BENCHMARK'){throw 'Only declared Development captures supported'}
if((Get-FileHash -LiteralPath (Join-Path $captureRoot 'raw-manifest.json') -Algorithm SHA256).Hash.ToLower() -ne $check.manifest_sha256){throw 'Manifest changed since verification'}
if((Get-FileHash -LiteralPath (Join-Path $captureRoot 'format-receipt.json') -Algorithm SHA256).Hash.ToLower() -ne $check.original_transport_receipt_sha256){throw 'Transport receipt changed since verification'}
if(Test-Path -LiteralPath $Receipt){throw 'Fresh deletion receipt required'}
$names=@('depth_left.transport.npy','depth_right.transport.npy','normal_left.transport.npy','albedo_left.transport.npy','isolated_depth_1.transport.npy','isolated_depth_254.transport.npy')
$framesRoot=Join-Path $captureRoot 'frames'
# Check every resolved target and each task-directory ancestor before any removal.
foreach($row in $check.delete_candidates){
    $path=[IO.Path]::GetFullPath($row.path)
    if(-not $path.StartsWith($framesRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Deletion target escaped frame tree'}
    if([IO.Path]::GetFileName($path) -notin $names){throw 'Unexpected deletion target'}
    $parent=Split-Path $path -Parent
    if((Split-Path $parent -Parent) -ne $framesRoot -or (Split-Path $parent -Leaf) -notmatch '^frame-\d{4,}$'){throw 'Unexpected frame directory'}
    foreach($itemPath in @($captureRoot,$framesRoot,$parent,$path)){
        $item=Get-Item -LiteralPath $itemPath
        if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse point in deletion tree'}
    }
    if((Get-Item -LiteralPath $path).Length -ne $row.bytes){throw 'Verified payload size changed'}
}
$result=@{schema='cnh-street-storage-release-v1';status='RUNNING';capture=$captureRoot;verification=(Resolve-Path -LiteralPath $Verification).Path;verification_sha256=(Get-FileHash -LiteralPath $Verification -Algorithm SHA256).Hash.ToLower();files_deleted=0;logical_bytes_removed=0L;errors=@();original_format_receipt='PRESERVED_HISTORICAL_HASHES';label_precision='NOT_INFERRED_FROM_ENCODING_CHECK'}
try{
    foreach($row in $check.delete_candidates){
        Remove-Item -LiteralPath $row.path -ErrorAction Stop
        if(Test-Path -LiteralPath $row.path){throw 'File survived removal'}
        $result.files_deleted++
        $result.logical_bytes_removed+=[long]$row.bytes
        if($result.files_deleted % 100 -eq 0){$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Receipt -Encoding utf8}
    }
    $remaining=@($check.delete_candidates | Where-Object {Test-Path -LiteralPath $_.path})
    if($remaining.Count){throw 'Removal verification failed'}
    $result.status='PASS_VERIFIED_CANONICAL_DUPLICATES_REMOVED'
}catch{
    $result.status='FAILED_PARTIAL_REMOVAL';$result.errors+= $_.Exception.Message
    throw
}finally{
    $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Receipt -Encoding utf8
}
$result | ConvertTo-Json -Depth 12
