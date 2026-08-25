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

& (Join-Path $target 'tools/ba.ps1') setup research-r1cl
if ($LASTEXITCODE -ne 0) { throw 'R1C-L worktree setup failed; worktree was preserved for diagnosis' }
Write-Output "PASS worktree=$target branch=$Branch"
