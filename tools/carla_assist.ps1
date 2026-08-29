[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('check', 'status', 'list', 'verify', 'explain', 'next', 'materialize')]
    [string]$Command = 'check',
    [Parameter(Position = 1)]
    [string]$Asset,
    [switch]$Deep,
    [string]$CarlaRoot,
    [string]$CarlaPython,
    [string]$Manifest = 'research/active/dtr-r0/carla/consumer-manifest.json',
    [string]$Output = 'artifacts.local/evidence/dtr-r0/carla/asset-context.json',
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$LocalConfigPath = Join-Path $RepoRoot 'config/local.toml'

function Read-LocalConfig {
    $values = @{}
    if (-not (Test-Path -LiteralPath $LocalConfigPath -PathType Leaf)) { return $values }
    foreach ($line in Get-Content -LiteralPath $LocalConfigPath) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed.StartsWith('[')) { continue }
        if ($trimmed -match '^([A-Za-z0-9_-]+)\s*=\s*"(.*)"\s*$') {
            $values[$Matches[1]] = $Matches[2]
        }
    }
    return $values
}

function Resolve-LocalPath {
    param([string]$Value, [string]$Base = $RepoRoot)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Base $Value))
}

function Resolve-ContainedPath {
    param([string]$Root, [string]$RelativePath, [string]$Label)
    if ([string]::IsNullOrWhiteSpace($RelativePath) -or [System.IO.Path]::IsPathRooted($RelativePath)) {
        throw "$Label must be a non-empty path relative to the CARLA root."
    }
    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $rootFull $RelativePath))
    $prefix = $rootFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label escapes the CARLA root: $RelativePath"
    }
    return $candidate
}

function Read-JsonFile {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is unavailable: $Path"
    }
    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable -Depth 100
    }
    catch {
        throw "$Label is not valid JSON: $Path ($($_.Exception.Message))"
    }
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Invoke-CarlaJson {
    param([string[]]$ToolArguments)
    $lines = @(& $script:CarlaPythonPath $script:AssetToolPath @ToolArguments)
    $exitCode = $LASTEXITCODE
    $raw = $lines -join "`n"
    if ($exitCode -ne 0) {
        throw "CARLA asset command failed (exit $exitCode): $($ToolArguments -join ' ')"
    }
    try {
        return $raw | ConvertFrom-Json -AsHashtable -Depth 100
    }
    catch {
        throw "CARLA asset command returned invalid JSON: $($ToolArguments -join ' ')"
    }
}

function Get-BridgeContext {
    param([switch]$DeepVerify)

    $manifestPath = Resolve-LocalPath $Manifest
    $consumer = Read-JsonFile $manifestPath 'BlindAssist CARLA consumer manifest'
    if ($consumer['schema_version'] -ne 'blindassist-carla-consumer-v1') {
        throw "Unsupported BlindAssist CARLA consumer schema: $($consumer['schema_version'])"
    }
    if ($consumer['consumer'] -ne 'blindassist-dtr-r0') {
        throw "Unexpected CARLA consumer: $($consumer['consumer'])"
    }
    if ($consumer['runtime']['bridge_mode'] -ne 'external_process_boundary') {
        throw "CARLA bridge_mode must be external_process_boundary."
    }

    $catalog = Read-JsonFile $script:CatalogPath 'CARLA asset catalog'
    $profile = Read-JsonFile $script:ProfilePath 'CARLA operational profile'
    if ($catalog['schema_version'] -ne 1) {
        throw "Unsupported CARLA asset catalog schema: $($catalog['schema_version'])"
    }
    $requiredVersion = [string]$consumer['runtime']['required_version']
    $catalogVersion = [string]$catalog['runtime']['version']
    $profileVersion = [string]$profile['runtime']['carla_version']
    if ($requiredVersion -ne $catalogVersion -or $requiredVersion -ne $profileVersion) {
        throw "CARLA version mismatch: required=$requiredVersion catalog=$catalogVersion profile=$profileVersion"
    }

    $catalogById = @{}
    foreach ($catalogAsset in @($catalog['assets'])) {
        $catalogById[[string]$catalogAsset['id']] = $catalogAsset
    }

    $requiredAssets = @($consumer['required_assets'])
    if ($requiredAssets.Count -eq 0) {
        throw 'The BlindAssist CARLA consumer manifest has no required assets.'
    }
    $seen = @{}
    $assetReports = @()
    foreach ($requiredAsset in $requiredAssets) {
        $assetId = [string]$requiredAsset['id']
        if ([string]::IsNullOrWhiteSpace($assetId)) { throw 'A required CARLA asset has no id.' }
        if ($seen.ContainsKey($assetId)) { throw "Duplicate required CARLA asset: $assetId" }
        $seen[$assetId] = $true
        if (-not $catalogById.ContainsKey($assetId)) { throw "Required CARLA asset is absent: $assetId" }

        $catalogAsset = $catalogById[$assetId]
        $expectedStatus = [string]$requiredAsset['expected_status']
        $expectedAuthority = [string]$requiredAsset['expected_authority']
        if ([string]$catalogAsset['status'] -ne $expectedStatus) {
            throw "CARLA status drift for ${assetId}: expected=$expectedStatus actual=$($catalogAsset['status'])"
        }
        if ([string]$catalogAsset['authority'] -ne $expectedAuthority) {
            throw "CARLA authority drift for ${assetId}: expected=$expectedAuthority actual=$($catalogAsset['authority'])"
        }

        $resultArtifacts = @($catalogAsset['artifacts'] | Where-Object { $_['role'] -eq 'result' })
        if ($resultArtifacts.Count -ne 1) { throw "CARLA asset $assetId must expose exactly one result artifact." }
        $catalogResult = $resultArtifacts[0]
        $expectedResult = $requiredAsset['result']
        $expectedRelativePath = ([string]$expectedResult['path']).Replace('\', '/')
        $catalogRelativePath = ([string]$catalogResult['path']).Replace('\', '/')
        $expectedSha = ([string]$expectedResult['sha256']).ToUpperInvariant()
        $catalogSha = ([string]$catalogResult['sha256']).ToUpperInvariant()
        if ($expectedRelativePath -ne $catalogRelativePath) {
            throw "CARLA result path drift for $assetId."
        }
        if ($expectedSha -ne $catalogSha) {
            throw "CARLA result catalog hash drift for $assetId."
        }
        $resultPath = Resolve-ContainedPath $script:CarlaRootPath $expectedRelativePath "$assetId result"
        if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
            throw "CARLA result is unavailable for ${assetId}: $resultPath"
        }
        $actualSha = Get-Sha256 $resultPath
        if ($actualSha -ne $expectedSha) {
            throw "CARLA result payload hash drift for ${assetId}: expected=$expectedSha actual=$actualSha"
        }

        $verifyArguments = @('verify', $assetId)
        if ($DeepVerify) { $verifyArguments += '--deep' }
        $verifyArguments += '--json'
        $verification = Invoke-CarlaJson $verifyArguments
        if (-not [bool]$verification['ok']) {
            throw "CARLA asset verification failed for $assetId."
        }

        $assetReports += [ordered]@{
            id = $assetId
            status = $expectedStatus
            authority = $expectedAuthority
            blindassist_role = [string]$requiredAsset['blindassist_role']
            result_relative_path = $expectedRelativePath
            result_sha256 = $actualSha
            result_absolute_path = $resultPath
            checked_files = [int]$verification['assets'][0]['checked_files']
            deep_files = [int]$verification['assets'][0]['deep_files']
            claim_ceiling = @($catalogAsset['claim_ceiling'])
        }
    }

    $status = Invoke-CarlaJson @('status', '--json')
    return [ordered]@{
        schema_version = 'blindassist-carla-asset-context-v1'
        generated_at = [System.DateTimeOffset]::UtcNow.ToString('o')
        consumer = [string]$consumer['consumer']
        bridge_mode = [string]$consumer['runtime']['bridge_mode']
        consumer_manifest = [ordered]@{
            path = $manifestPath
            sha256 = Get-Sha256 $manifestPath
        }
        external = [ordered]@{
            root = $script:CarlaRootPath
            python = $script:CarlaPythonPath
            runtime_version = $requiredVersion
            profile_id = [string]$profile['profile_id']
            catalog_sha256 = Get-Sha256 $script:CatalogPath
            profile_sha256 = Get-Sha256 $script:ProfilePath
            reliability_status = [string]$status['reliability_status']
            ports = $status['ports']
        }
        assets = $assetReports
        claim_boundary = @($consumer['claim_boundary'])
        external_next_route = $profile['next_route']
        blindassist_next_target = $consumer['next_target']
    }
}

try {
    $local = Read-LocalConfig
    $rootValue = $CarlaRoot
    if ([string]::IsNullOrWhiteSpace($rootValue) -and $local.ContainsKey('carla_root')) {
        $rootValue = $local['carla_root']
    }
    if ([string]::IsNullOrWhiteSpace($rootValue)) {
        $rootValue = $env:BLINDASSIST_CARLA_ROOT
    }
    if ([string]::IsNullOrWhiteSpace($rootValue)) {
        $rootValue = Join-Path $RepoRoot '..\CARLA'
    }
    $script:CarlaRootPath = Resolve-LocalPath $rootValue
    $script:CatalogPath = Join-Path $script:CarlaRootPath 'asset-catalog.json'
    $script:ProfilePath = Join-Path $script:CarlaRootPath 'operational-profile.json'
    $script:AssetToolPath = Join-Path $script:CarlaRootPath 'tools/carla_asset.py'
    foreach ($requiredPath in @($script:CatalogPath, $script:ProfilePath, $script:AssetToolPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "CARLA external asset library is incomplete: $requiredPath"
        }
    }

    $pythonValue = $CarlaPython
    if ([string]::IsNullOrWhiteSpace($pythonValue) -and $local.ContainsKey('carla_python')) {
        $pythonValue = $local['carla_python']
    }
    if ([string]::IsNullOrWhiteSpace($pythonValue)) {
        $pythonValue = $env:BLINDASSIST_CARLA_PYTHON
    }
    if ([string]::IsNullOrWhiteSpace($pythonValue)) {
        $pythonValue = 'client-env/Scripts/python.exe'
    }
    $script:CarlaPythonPath = Resolve-LocalPath $pythonValue $script:CarlaRootPath
    if (-not (Test-Path -LiteralPath $script:CarlaPythonPath -PathType Leaf)) {
        throw "CARLA Python is unavailable: $script:CarlaPythonPath"
    }

    if ($Command -notin @('check', 'materialize')) {
        $forward = @($Command)
        if ($Command -eq 'verify') {
            $forward += $(if ([string]::IsNullOrWhiteSpace($Asset)) { 'all' } else { $Asset })
            if ($Deep) { $forward += '--deep' }
        }
        elseif ($Command -eq 'explain') {
            if ([string]::IsNullOrWhiteSpace($Asset)) { throw 'explain requires -Asset <asset-id>.' }
            $forward += $Asset
        }
        if ($Json) { $forward += '--json' }
        & $script:CarlaPythonPath $script:AssetToolPath @forward
        exit $LASTEXITCODE
    }

    $context = Get-BridgeContext -DeepVerify:$Deep
    if ($Command -eq 'check') {
        if ($Json) {
            $context | ConvertTo-Json -Depth 100
        }
        else {
            Write-Output 'PASS blindassist-carla external asset bridge'
            Write-Output "consumer_manifest: $($context['consumer_manifest']['path'])"
            Write-Output "carla_root: $($context['external']['root'])"
            Write-Output "carla_python: $($context['external']['python'])"
            Write-Output "runtime: $($context['external']['runtime_version'])"
            foreach ($item in @($context['assets'])) {
                Write-Output "PASS $($item['id']) status=$($item['status']) metadata=$($item['checked_files']) deep=$($item['deep_files'])"
            }
            $portText = @($context['external']['ports'].GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ', '
            Write-Output "ports: $portText"
        }
        exit 0
    }

    $outputPath = Resolve-LocalPath $Output
    $outputDirectory = Split-Path -Parent $outputPath
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
    $serialized = $context | ConvertTo-Json -Depth 100
    [System.IO.File]::WriteAllText(
        $outputPath,
        $serialized + [System.Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    $receipt = [ordered]@{
        status = 'MATERIALIZED'
        output = $outputPath
        sha256 = Get-Sha256 $outputPath
        assets = @($context['assets']).Count
    }
    if ($Json) { $receipt | ConvertTo-Json -Depth 10 }
    else {
        Write-Output "MATERIALIZED $outputPath"
        Write-Output "sha256: $($receipt['sha256'])"
        Write-Output "assets: $($receipt['assets'])"
    }
    exit 0
}
catch {
    [Console]::Error.WriteLine("CARLA_ASSIST_ERROR: $($_.Exception.Message)")
    exit 2
}
