[CmdletBinding()]
param(
    [string]$Python = 'E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe',
    [string]$RuntimeRoot = 'artifacts.local\runtime\semantic-anchor-v1'
)

$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirements = Join-Path $packageRoot 'requirements-semantic-anchor-v1.txt'
$sitePackages = Join-Path $RuntimeRoot 'site-packages'
New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null

& $Python -m pip install --target $sitePackages --upgrade --no-deps -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Semantic-anchor runtime installation failed with exit code $LASTEXITCODE"
}

$resolvedSitePackages = (Resolve-Path $sitePackages).Path
$env:PYTHONPATH = $resolvedSitePackages
& $Python -c "from rapidocr import RapidOCR; import cv2, onnxruntime; assert hasattr(cv2, 'aruco'); print('semantic_anchor_runtime=ready'); print('onnxruntime=' + onnxruntime.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Semantic-anchor runtime verification failed with exit code $LASTEXITCODE"
}
