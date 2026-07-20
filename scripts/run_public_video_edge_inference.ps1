# Stable Adapter: keep callers independent from the archived campaign layout.
$implementation = Join-Path $PSScriptRoot 'research/public_video/run_public_video_edge_inference.ps1'
& $implementation @args
if ($null -ne $LASTEXITCODE) {
    exit $LASTEXITCODE
}
