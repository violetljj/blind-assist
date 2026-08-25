[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Path,
    [string]$Branch = 'codex/grail-r1cl'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$target = [System.IO.Path]::GetFullPath($Path)
if (Test-Path -LiteralPath $target) { throw "Target already exists: $target" }

$env:GIT_LFS_SKIP_SMUDGE = '1'
git -C $repo worktree add -b $Branch $target master
if ($LASTEXITCODE -ne 0) { throw 'git worktree add failed' }

foreach ($relative in @('config/local.toml', 'local.properties')) {
    $source = Join-Path $repo $relative
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        $destination = Join-Path $target $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }
}

# A copied local config must keep pointing at machine-owned payloads beside the
# source checkout. Relative values would otherwise be reinterpreted inside the
# new worktree, which intentionally contains no datasets or model weights.
$targetConfig = Join-Path $target 'config/local.toml'
if (Test-Path -LiteralPath $targetConfig -PathType Leaf) {
    $pathKeys = @(
        'research_python', 'r1cl_backbone', 'r1cl_train_dataset',
        'r1cl_validation_dataset', 'r1cl_output', 'android_sdk', 'java_home',
        'adb', 'docker', 'export_python'
    )
    $rebased = foreach ($line in Get-Content -LiteralPath $targetConfig) {
        if ($line -match '^([A-Za-z0-9_-]+)\s*=\s*"(.*)"\s*$' -and
            $Matches[1] -in $pathKeys -and $Matches[2] -and
            -not [IO.Path]::IsPathRooted($Matches[2])) {
            $absolute = [IO.Path]::GetFullPath((Join-Path $repo $Matches[2])).Replace('\', '/')
            "$($Matches[1]) = `"$absolute`""
        }
        else { $line }
    }
    Set-Content -LiteralPath $targetConfig -Value $rebased -Encoding utf8NoBOM
}

& (Join-Path $target 'tools/ba.ps1') setup research-r1cl
if ($LASTEXITCODE -ne 0) { throw 'R1C-L worktree setup failed; worktree was preserved for diagnosis' }
Write-Output "PASS worktree=$target branch=$Branch"
