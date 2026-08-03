param(
    [Parameter(Mandatory = $true)]
    [string[]] $Manifest,

    [Parameter(Mandatory = $true)]
    [string] $OutputDir,

    [string] $Python = 'E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$outputRoot = [System.IO.Path]::GetFullPath($OutputDir)
$dependencyRoot = Join-Path $repoRoot 'artifacts.local\vendor\python-packages-hftf-metric-depth-r0'
$env:PYTHONPATH = $dependencyRoot
$env:HF_HOME = Join-Path $repoRoot 'artifacts.local\models\hftf-external-rgb-metric-track-r0\huggingface'

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$producer = Join-Path $PSScriptRoot 'produce_external_rgb_metric_depth_observations.py'
$evaluator = Join-Path $PSScriptRoot 'evaluate_external_rgb_metric_depth_source.py'
$resolvedManifests = @($Manifest | ForEach-Object { (Resolve-Path $_).Path })

$unidepthOutput = Join-Path $outputRoot 'unidepth-v2-vits14.jsonl'
$vdaOutput = Join-Path $outputRoot 'video-depth-anything-metric-vits-stream.jsonl'
$metric3dOutput = Join-Path $outputRoot 'metric3d-v2-vits-pytorch.jsonl'
$reportOutput = Join-Path $outputRoot 'triarm-report.json'

$common = @($producer, '--manifest') + $resolvedManifests

& $Python @common `
    --model 'unidepth-v2-vits14' `
    --output $unidepthOutput `
    --unidepth-repo (Join-Path $repoRoot 'artifacts.local\vendor\UniDepth') `
    --device 'cuda' `
    --unidepth-resolution-level 0
if ($LASTEXITCODE -ne 0) { throw "UniDepth failed with exit code $LASTEXITCODE" }

& $Python @common `
    --model 'video-depth-anything-metric-vits-stream' `
    --output $vdaOutput `
    --vda-repo (Join-Path $repoRoot 'artifacts.local\vendor\Video-Depth-Anything') `
    --vda-checkpoint (Join-Path $repoRoot 'artifacts.local\vendor\Video-Depth-Anything\checkpoints\metric_video_depth_anything_vits.pth') `
    --device 'cuda' `
    --vda-input-size 392
if ($LASTEXITCODE -ne 0) { throw "Video Depth Anything failed with exit code $LASTEXITCODE" }

& $Python @common `
    --model 'metric3d-v2-vits-pytorch' `
    --output $metric3dOutput `
    --metric3d-repo (Join-Path $repoRoot 'artifacts.local\vendor\Metric3D') `
    --metric3d-checkpoint (Join-Path $repoRoot 'artifacts.local\models\hftf-external-rgb-metric-track-r0\metric3d-vit-small-pytorch\metric_depth_vit_small_800k.pth') `
    --device 'cuda'
if ($LASTEXITCODE -ne 0) { throw "Metric3D failed with exit code $LASTEXITCODE" }

& $Python $evaluator `
    --observations $unidepthOutput $vdaOutput $metric3dOutput `
    --output $reportOutput
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }

Write-Output $reportOutput
