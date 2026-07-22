[CmdletBinding()]
param(
    [string]$StructureScript = (Join-Path $PSScriptRoot 'check_project_structure.ps1')
)

$ErrorActionPreference = 'Stop'

function Write-TestFile([string]$Repository, [string]$RelativePath, [string]$Content) {
    $path = Join-Path $Repository $RelativePath
    $parent = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    [IO.File]::WriteAllText($path, $Content, [Text.UTF8Encoding]::new($false))
}

function Research-Readme([string]$Name) {
    return @"
# $Name

状态：active

## 稳定 Interface
Use the root Adapter.

## 输出
All outputs stay under artifacts.local/.

## 安全边界
Research-only.

## 停止条件
Stop after the frozen acceptance condition fails.
"@
}

function New-TestRepository([string]$Name) {
    $repository = Join-Path $script:TestRoot $Name
    New-Item -ItemType Directory -Path $repository | Out-Null
    & git -C $repository init --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to initialize test repository: $repository"
    }

    Write-TestFile $repository 'scripts/README.md' '# scripts'
    Write-TestFile $repository 'scripts/check.ps1' 'Write-Host ok'
    Write-TestFile $repository 'scripts/research/demo/README.md' (Research-Readme 'demo')
    Write-TestFile $repository 'scripts/research/demo/tool.py' 'from research.common.util import value'
    Write-TestFile $repository 'scripts/research/common/README.md' (Research-Readme 'common')
    Write-TestFile $repository 'scripts/research/common/util.py' 'value = 1'
    Write-TestFile $repository 'AGENTS.md' 'Default to an end-to-end autonomous workflow. Do not create, preserve, or wait on a human-required queue or gate. Data available through an ordinary public channel may enter isolated internal research.'
    Write-TestFile $repository 'docs/AI_REVIEW_GOVERNANCE.md' '项目默认端到端无人化。当前 Codex/GPT 会话就是执行模型，不要求 API key；不建立、不保留、也不等待人工队列。'
    Write-TestFile $repository 'configs/ai_review_workflows_v1.json' '{"execution_surface":"current_codex_or_gpt_session","human_required_queue_forbidden":true,"ordinary_public_download_is_sufficient_for_isolated_internal_research":true}'
    Write-TestFile $repository 'DEVELOPMENT_LOG.md' "# Development Log`n`n## 2026-07-21`n`n### fixture`n- 时间：2026-07-21；执行者：test。`n"
    Write-TestFile $repository 'policy/root-files.txt' "README.md`ncheck.ps1`n"

    $policy = [ordered]@{
        schema_version = 1
        scripts_root = 'scripts'
        root_allowlist_path = 'policy/root-files.txt'
        root_index_exempt_patterns = @('^test_', '^README\.md$')
        development_log = [ordered]@{
            path = 'DEVELOPMENT_LOG.md'
            max_lines = 20
            max_bytes = 4096
            max_age_days = 28
        }
        research_root = 'scripts/research'
        research_readme_required_markers = @('状态：', '## 稳定 Interface', '## 输出', '## 安全边界', '## 停止条件', 'artifacts.local/')
        internal_reference_source_allowlist = @()
    }
    Write-TestFile $repository 'policy/project_structure.json' ($policy | ConvertTo-Json -Depth 5)
    $authorityPolicy = [ordered]@{
        version = 1
        scan_roots = @('AGENTS.md', 'configs', 'scripts', 'docs')
        scan_extensions = @('.md', '.json', '.py', '.ps1')
        exclude_path_prefixes = @('scripts/policy/ai_review_authority.json')
        scan_paths = @()
        forbidden_patterns = @('reviewer_type[\s\"'':=]+human', 'waiting_for_human', 'human_operator_required[\s\"'':=]+true')
        required_markers = [ordered]@{
            'AGENTS.md' = @('Default to an end-to-end autonomous workflow', 'Do not create, preserve, or wait on a human-required queue or gate', 'ordinary public channel')
            'docs/AI_REVIEW_GOVERNANCE.md' = @('项目默认端到端无人化', '当前 Codex/GPT 会话就是执行模型', '不要求 API key', '不建立、不保留、也不等待人工队列')
            'configs/ai_review_workflows_v1.json' = @('current_codex_or_gpt_session', 'human_required_queue_forbidden', 'ordinary_public_download_is_sufficient_for_isolated_internal_research')
        }
    }
    Write-TestFile $repository 'scripts/policy/ai_review_authority.json' ($authorityPolicy | ConvertTo-Json -Depth 6)
    return $repository
}

function Invoke-StructureCheck([string]$Repository, [string]$BaseRef = '') {
    & $script:StructureScript `
        -RepoRoot $Repository `
        -PolicyPath (Join-Path $Repository 'policy/project_structure.json') `
        -BaseRef $BaseRef `
        -AsOfDate ([datetime]'2026-07-21')
    return $LASTEXITCODE
}

function Assert-Scenario([string]$Name, [bool]$ShouldPass, [scriptblock]$Mutate = {}) {
    $repository = New-TestRepository $Name
    & $Mutate $repository
    $exitCode = Invoke-StructureCheck $repository
    $passed = $exitCode -eq 0
    if ($passed -ne $ShouldPass) {
        $expectation = if ($ShouldPass) { 'pass' } else { 'fail' }
        throw "Scenario '$Name' was expected to $expectation but exited with code $exitCode."
    }
    Write-Host "PASS: $Name"
}

$script:StructureScript = (Resolve-Path -LiteralPath $StructureScript).Path
$script:TestRoot = Join-Path ([IO.Path]::GetTempPath()) ("blindassist-project-structure-{0}" -f [guid]::NewGuid().ToString('N'))

try {
    New-Item -ItemType Directory -Path $script:TestRoot | Out-Null

    Assert-Scenario 'valid' $true
    Assert-Scenario 'root-file-rejected' $false {
        param($repo)
        Write-TestFile $repo 'scripts/experiment_r99.py' 'print("no")'
    }
    Assert-Scenario 'root-allowlist-stale' $false {
        param($repo)
        Remove-Item -LiteralPath (Join-Path $repo 'scripts/check.ps1') -Force
    }
    Assert-Scenario 'log-line-budget' $false {
        param($repo)
        $lines = @('# Development Log', '', '## 2026-07-21') + (1..25 | ForEach-Object { "- line $_" })
        Write-TestFile $repo 'DEVELOPMENT_LOG.md' ($lines -join "`n")
    }
    Assert-Scenario 'log-retention' $false {
        param($repo)
        Write-TestFile $repo 'DEVELOPMENT_LOG.md' "# Development Log`n`n## 2026-06-01`n- 时间：2026-06-01`n"
    }
    Assert-Scenario 'log-budget-cannot-be-raised' $false {
        param($repo)
        $policyPath = Join-Path $repo 'policy/project_structure.json'
        $policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $policy.development_log.max_lines = 999999
        Write-TestFile $repo 'policy/project_structure.json' ($policy | ConvertTo-Json -Depth 5)
    }
    Assert-Scenario 'missing-module-readme' $false {
        param($repo)
        Remove-Item -LiteralPath (Join-Path $repo 'scripts/research/demo/README.md') -Force
    }
    Assert-Scenario 'incomplete-module-contract' $false {
        param($repo)
        Write-TestFile $repo 'scripts/research/demo/README.md' "# demo`n状态：active`n"
    }
    Assert-Scenario 'private-path-reference' $false {
        param($repo)
        Write-TestFile $repo 'docs/caller.md' 'python scripts/research/demo/tool.py'
    }
    Assert-Scenario 'cross-module-private-import' $false {
        param($repo)
        Write-TestFile $repo 'scripts/research/other/README.md' (Research-Readme 'other')
        Write-TestFile $repo 'scripts/research/other/tool.py' 'from research.demo.tool import main'
    }
    Assert-Scenario 'human-gate-in-current-doc' $false {
        param($repo)
        Write-TestFile $repo 'docs/current.md' 'status: waiting_for_human_review'
    }
    Assert-Scenario 'human-gate-in-script' $false {
        param($repo)
        Write-TestFile $repo 'scripts/research/demo/human_gate.py' 'reviewer_type = "human"'
    }
    Assert-Scenario 'human-truth-disclaimer-is-not-a-gate' $true {
        param($repo)
        Write-TestFile $repo 'docs/current.md' 'Model evidence is not human truth or objective sensor measurement.'
    }

    $indexedRepository = New-TestRepository 'new-root-interface-index'
    & git -C $indexedRepository add --all
    & git -C $indexedRepository -c user.name=structure-test -c user.email=structure-test@example.invalid commit --quiet -m baseline
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create structure-test baseline commit.' }
    $baseCommit = (& git -C $indexedRepository rev-parse HEAD).Trim()
    Write-TestFile $indexedRepository 'scripts/new_stable.ps1' 'Write-Host stable'
    Write-TestFile $indexedRepository 'policy/root-files.txt' "README.md`ncheck.ps1`nnew_stable.ps1`n"
    if ((Invoke-StructureCheck $indexedRepository $baseCommit) -eq 0) {
        throw 'New root Interface without scripts/README.md entry was expected to fail.'
    }
    Write-TestFile $indexedRepository 'scripts/README.md' "# scripts`n- new_stable.ps1`n"
    if ((Invoke-StructureCheck $indexedRepository $baseCommit) -ne 0) {
        throw 'Indexed new root Interface was expected to pass.'
    }
    Write-Host 'PASS: new-root-interface-index'

    $testExemptRepository = New-TestRepository 'new-root-test-index-exempt'
    & git -C $testExemptRepository add --all
    & git -C $testExemptRepository -c user.name=structure-test -c user.email=structure-test@example.invalid commit --quiet -m baseline
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create test-exempt baseline commit.' }
    $testExemptBase = (& git -C $testExemptRepository rev-parse HEAD).Trim()
    Write-TestFile $testExemptRepository 'scripts/test_new_stable.py' 'print("test")'
    Write-TestFile $testExemptRepository 'policy/root-files.txt' "README.md`ncheck.ps1`ntest_new_stable.py`n"
    if ((Invoke-StructureCheck $testExemptRepository $testExemptBase) -ne 0) {
        throw 'New root test matching the policy exemption was expected to pass without a stable-index entry.'
    }
    Write-Host 'PASS: new-root-test-index-exempt'

    $bootstrapRepository = New-TestRepository 'gate-bootstrap-base'
    Remove-Item -LiteralPath (Join-Path $bootstrapRepository 'policy/project_structure.json') -Force
    & git -C $bootstrapRepository add --all
    & git -C $bootstrapRepository -c user.name=structure-test -c user.email=structure-test@example.invalid commit --quiet -m pre-gate-baseline
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create pre-gate baseline commit.' }
    $bootstrapBase = (& git -C $bootstrapRepository rev-parse HEAD).Trim()
    $bootstrapPolicy = [ordered]@{
        schema_version = 1
        scripts_root = 'scripts'
        root_allowlist_path = 'policy/root-files.txt'
        root_index_exempt_patterns = @('^test_', '^README\.md$')
        development_log = [ordered]@{ path = 'DEVELOPMENT_LOG.md'; max_lines = 20; max_bytes = 4096; max_age_days = 28 }
        research_root = 'scripts/research'
        research_readme_required_markers = @('状态：', '## 稳定 Interface', '## 输出', '## 安全边界', '## 停止条件', 'artifacts.local/')
        internal_reference_source_allowlist = @()
    }
    Write-TestFile $bootstrapRepository 'policy/project_structure.json' ($bootstrapPolicy | ConvertTo-Json -Depth 5)
    Write-TestFile $bootstrapRepository 'scripts/preexisting_tool.py' 'print("pre-gate")'
    Write-TestFile $bootstrapRepository 'policy/root-files.txt' "README.md`ncheck.ps1`npreexisting_tool.py`n"
    if ((Invoke-StructureCheck $bootstrapRepository $bootstrapBase) -ne 0) {
        throw 'Gate bootstrap against a base without the policy was expected to skip retroactive index enforcement.'
    }
    Write-Host 'PASS: gate-bootstrap-base'

    Write-Host 'Project structure smoke tests passed.'
}
finally {
    if (Test-Path -LiteralPath $script:TestRoot) {
        $resolvedTestRoot = [IO.Path]::GetFullPath($script:TestRoot)
        $resolvedTempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if (-not $resolvedTestRoot.StartsWith($resolvedTempRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove test path outside temp: $resolvedTestRoot"
        }
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
