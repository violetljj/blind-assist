[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$IfChanged,
    [switch]$Staged,
    [switch]$StageGenerated,
    [switch]$InstallHook,
    [string]$PythonCommand = 'python'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel 2>$null | Select-Object -First 1).Trim()
if (-not $repoRoot) { throw 'Could not locate the BlindAssist Git root.' }

function Test-KnowledgePath([string]$Path) {
    $normalized = $Path.Replace('\', '/')
    return (
        $normalized.StartsWith('research/knowledge/') -or
        $normalized -eq 'experiments/index.jsonl' -or
        $normalized -eq 'tools/knowledge.py' -or
        $normalized -eq 'tools/test_knowledge.py' -or
        $normalized -eq 'tools/migrate_scattered_knowledge.py' -or
        $normalized -eq 'scripts/refresh_knowledge.ps1' -or
        $normalized -eq '.githooks/pre-commit'
    )
}

function Invoke-PythonStep([string]$Label, [string[]]$CommandArgs) {
    Write-Host "KNOWLEDGE $Label"
    & $PythonCommand @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Knowledge refresh step failed: $Label"
    }
}

if ($Check -and $StageGenerated) {
    throw '-Check and -StageGenerated cannot be used together.'
}

if ($InstallHook) {
    & git -C $repoRoot config core.hooksPath .githooks
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the local Git hook path.' }
    Write-Host 'KNOWLEDGE HOOK installed: core.hooksPath=.githooks'
}

if ($IfChanged) {
    $changedPaths = if ($Staged) {
        @(& git -C $repoRoot diff --cached --name-only --diff-filter=ACMR)
    } else {
        @(& git -C $repoRoot diff --name-only --diff-filter=ACMR)
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect changed paths.' }
    $relevantPaths = @($changedPaths | Where-Object { Test-KnowledgePath $_ })
    if (-not $relevantPaths) {
        Write-Host 'KNOWLEDGE SKIP: no relevant knowledge or experiment changes.'
        exit 0
    }
    Write-Host "KNOWLEDGE TRIGGER: $($relevantPaths -join ', ')"
}

if ($StageGenerated) {
    if (-not $Staged) {
        throw '-StageGenerated requires -Staged so unstaged source edits cannot leak.'
    }
    $unstaged = @(& git -C $repoRoot diff --name-only --diff-filter=ACMR)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect unstaged paths.' }
    $untracked = @(& git -C $repoRoot ls-files --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect untracked paths.' }
    $unsafe = @(
        @($unstaged + $untracked) |
            Sort-Object -Unique |
            Where-Object {
                (Test-KnowledgePath $_) -and
                $_.Replace('\', '/') -ne 'research/knowledge/decision/index.json'
            }
    )
    if ($unsafe) {
        throw (
            'Relevant unstaged files would contaminate the generated index: ' +
            ($unsafe -join ', ')
        )
    }
}

Push-Location $repoRoot
try {
    $buildArgs = @('tools/knowledge.py', 'build-decision-index')
    if ($Check) { $buildArgs += '--check' }
    Invoke-PythonStep 'INDEX' $buildArgs
    Invoke-PythonStep 'UNIT' @('-m', 'unittest', 'tools.test_knowledge')
    Invoke-PythonStep 'VALIDATE' @('tools/knowledge.py', 'validate')
    Invoke-PythonStep 'DECISION' @('tools/knowledge.py', 'evaluate-decision-engine')
    if ($StageGenerated) {
        & git add -- research/knowledge/decision/index.json
        if ($LASTEXITCODE -ne 0) { throw 'Could not stage the refreshed decision index.' }
        Write-Host 'KNOWLEDGE STAGED: research/knowledge/decision/index.json'
    }
    Write-Host "PASS knowledge refresh (check=$Check)"
} finally {
    Pop-Location
}
