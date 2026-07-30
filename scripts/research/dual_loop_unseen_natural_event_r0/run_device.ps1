[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "Build",
        "Install",
        "PrepareInput",
        "RunBaseline",
        "CollectBaseline",
        "EvaluateBaseline",
        "StageAuthorization",
        "RunCandidate",
        "CollectCandidate",
        "EvaluateCandidate"
    )]
    [string]$Action,
    [string]$DeviceSerial = "R5CX10M8Y8X"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$adb = $null
$package = "com.linnan.blindassist"
$instrumentation =
    "com.linnan.blindassist.benchmark/androidx.test.runner.AndroidJUnitRunner"
$class =
    "com.linnan.blindassist.benchmark.ProductionTemporalGeometryFactorialAbDeviceTest"
$remoteBase =
    "/sdcard/Android/data/$package/files/dual_loop_production_temporal_ab_r0"
$remoteRun = "$remoteBase/unseen-natural-rank2-shiraz"
$evidenceRoot = Join-Path $repoRoot `
    "artifacts.local/evidence/dual-loop-r1-unseen-natural-event-r0/rank2-shiraz"
$inputRoot = Join-Path $evidenceRoot "input-10hz-r1"
$truthRoot = Join-Path $evidenceRoot "truth-freeze-r2"
$deviceRoot = Join-Path $evidenceRoot "device-r1"
$baselineRoot = Join-Path $deviceRoot "baseline-output"
$candidateRoot = Join-Path $deviceRoot "candidate-output"
$baselineEvaluationRoot = Join-Path $evidenceRoot "baseline-evaluation-r1"
$effectEvaluationRoot = Join-Path $evidenceRoot "effect-evaluation-r1"
$authorization = Join-Path $baselineEvaluationRoot "candidate_authorization.json"
$assessment = Join-Path $baselineEvaluationRoot "baseline_assessment.json"
$buildIdentity = Join-Path $evidenceRoot "build-identity-r1.json"
$candidateCommit = "039757b2da41c051373f8ee3189c4b06028f5295"
$protocolId = "DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_SHIRAZ"
$rank2ProtocolSha256 =
    "fe5862afce85c6d4e0f90891d61293f0c482c176b1d70f702c7f1da0e75098d9"
$rank2ProtocolPath = Join-Path $repoRoot `
    "docs/research/dual-loop/DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_PROTOCOL_2026-07-31.json"
$appApk = Join-Path $repoRoot "app/build/outputs/apk/debug/app-debug.apk"
$testApk = Join-Path $repoRoot `
    "device-benchmark/build/outputs/apk/debug/device-benchmark-debug.apk"

function Invoke-Adb {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $adb -s $DeviceSerial @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Assert-Device {
    $state = (& $adb -s $DeviceSerial get-state 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $state -ne "device") {
        throw "Required device is not connected: $DeviceSerial"
    }
    $model = (& $adb -s $DeviceSerial shell getprop ro.product.model).Trim()
    $soc = (& $adb -s $DeviceSerial shell getprop ro.soc.model).Trim()
    $abis = (& $adb -s $DeviceSerial shell getprop ro.product.cpu.abilist).Trim()
    if ($model -ne "SM-S9280" -or $soc -ne "SM8650" -or $abis -ne "arm64-v8a") {
        throw "Device identity mismatch: model=$model soc=$soc abis=$abis"
    }
}

function Get-InstalledApkSha256 {
    param([Parameter(Mandatory = $true)][string]$PackageName)
    $pathOutput = & $adb -s $DeviceSerial shell pm path $PackageName
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve installed APK: $PackageName"
    }
    $paths = @(
        $pathOutput |
            Where-Object { $_ -like "package:*" } |
            ForEach-Object { $_.Substring("package:".Length).Trim() }
    )
    if ($paths.Count -ne 1 -or -not $paths[0].EndsWith("/base.apk")) {
        throw "Unexpected installed APK path: $PackageName"
    }
    $hashOutput = & $adb -s $DeviceSerial shell sha256sum $paths[0]
    if ($LASTEXITCODE -ne 0) {
        throw "Could not hash installed APK: $PackageName"
    }
    $digest = ((($hashOutput -join "`n") -split "\s+")[0]).ToLowerInvariant()
    if ($digest -notmatch "^[0-9a-f]{64}$") {
        throw "Invalid installed APK SHA-256: $PackageName"
    }
    return $digest
}

function Assert-RemoteAbsent {
    param([Parameter(Mandatory = $true)][string]$Path)
    $state = (& $adb -s $DeviceSerial shell `
        "if [ -e '$Path' ]; then echo EXISTS; else echo ABSENT; fi").Trim()
    if ($state -ne "ABSENT") {
        throw "Remote path already exists; refusing to overwrite: $Path"
    }
}

function Invoke-Instrumentation {
    param([Parameter(Mandatory = $true)][string]$Method)
    $output = & $adb -s $DeviceSerial shell am instrument -w -r -e class `
        "$class#$Method" $instrumentation
    $text = $output -join "`n"
    $output | Write-Output
    if ($LASTEXITCODE -ne 0 -or $text -notmatch "OK \(1 test\)") {
        throw "Instrumentation did not complete successfully: $Method"
    }
}

function Collect-RemoteOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Remote,
        [Parameter(Mandatory = $true)][string]$Local
    )
    if (Test-Path -LiteralPath $Local) {
        throw "Host output already exists; refusing to overwrite: $Local"
    }
    $temporary = "$Local.tmp"
    if (Test-Path -LiteralPath $temporary) {
        throw "Stale host temporary output exists: $temporary"
    }
    New-Item -ItemType Directory -Force -Path $temporary | Out-Null
    try {
        Invoke-Adb -Arguments @("pull", "$Remote/trace.jsonl", "$temporary/")
        Invoke-Adb -Arguments @("pull", "$Remote/producer_receipt.json", "$temporary/")
        Move-Item -LiteralPath $temporary -Destination $Local
    } catch {
        throw
    }
}

$deviceActions = @(
    "Install",
    "PrepareInput",
    "RunBaseline",
    "CollectBaseline",
    "StageAuthorization",
    "RunCandidate",
    "CollectCandidate"
)
if ($Action -in $deviceActions) {
    $adb = (Get-Command adb -ErrorAction Stop).Source
    Assert-Device
}

switch ($Action) {
    "Build" {
        $gitStatus = & git -C $repoRoot status --short
        if ($LASTEXITCODE -ne 0 -or ($gitStatus -join "`n").Trim()) {
            throw "Rank-2 build requires a clean worktree"
        }
        $head = (& git -C $repoRoot rev-parse HEAD).Trim()
        $originMaster = (& git -C $repoRoot rev-parse origin/master).Trim()
        if ($LASTEXITCODE -ne 0 -or $head -ne $originMaster) {
            throw "Rank-2 build requires HEAD == origin/master"
        }
        & git -C $repoRoot diff --quiet $candidateCommit $head -- `
            core/assist core/vision app/src
        if ($LASTEXITCODE -ne 0) {
            throw "Frozen candidate runtime paths differ from $candidateCommit"
        }
        if (Test-Path -LiteralPath $buildIdentity) {
            throw "Build identity already exists; refusing to overwrite: $buildIdentity"
        }
        $actualProtocolSha = (
            Get-FileHash -LiteralPath $rank2ProtocolPath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        if ($actualProtocolSha -ne $rank2ProtocolSha256) {
            throw "Rank-2 protocol hash drift"
        }
        $env:JAVA_HOME = "E:\codex-tools\jdk-17"
        if (-not (Test-Path -LiteralPath $env:JAVA_HOME)) {
            $env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
        }
        & (Join-Path $repoRoot "gradlew.bat") `
            :app:assembleDebug :device-benchmark:assembleDebug --no-daemon
        if ($LASTEXITCODE -ne 0) {
            throw "Gradle Android build failed"
        }
        $identity = [ordered]@{
            schema_version = "blindassist.dual_loop_unseen_rank2_build_identity.v1"
            protocol_id = $protocolId
            status = "COMPLETE"
            candidate_commit = $candidateCommit
            runner_commit = $head
            origin_master_commit = $originMaster
            runtime_paths = @("core/assist", "core/vision", "app/src")
            runtime_path_diff_empty = $true
            rank2_protocol_sha256 = $actualProtocolSha
            app_apk_sha256 = (
                Get-FileHash -LiteralPath $appApk -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            test_apk_sha256 = (
                Get-FileHash -LiteralPath $testApk -Algorithm SHA256
            ).Hash.ToLowerInvariant()
        }
        New-Item -ItemType Directory -Force -Path `
            (Split-Path -Parent $buildIdentity) | Out-Null
        $temporary = "$buildIdentity.tmp"
        $identity | ConvertTo-Json -Depth 5 |
            Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
        Move-Item -LiteralPath $temporary -Destination $buildIdentity
    }
    "Install" {
        if (
            -not (Test-Path -LiteralPath $appApk) -or
            -not (Test-Path -LiteralPath $testApk)
        ) {
            throw "Build APKs before installation"
        }
        Invoke-Adb -Arguments @("install", "-r", $appApk)
        Invoke-Adb -Arguments @("install", "-r", $testApk)
        if (-not (Test-Path -LiteralPath $buildIdentity)) {
            throw "Build identity is missing: $buildIdentity"
        }
        $identity = Get-Content -LiteralPath $buildIdentity -Raw -Encoding UTF8 |
            ConvertFrom-Json
        if (
            (Get-InstalledApkSha256 -PackageName "com.linnan.blindassist") -ne
                $identity.app_apk_sha256 -or
            (
                Get-InstalledApkSha256 `
                    -PackageName "com.linnan.blindassist.benchmark"
            ) -ne $identity.test_apk_sha256
        ) {
            throw "Installed APK identity differs from build receipt"
        }
        $installed = & $adb -s $DeviceSerial shell pm list instrumentation
        if (($installed -join "`n") -notmatch [regex]::Escape($instrumentation)) {
            throw "Expected instrumentation is not installed"
        }
    }
    "PrepareInput" {
        foreach ($required in @(
            (Join-Path $inputRoot "input_receipt.json"),
            (Join-Path $inputRoot "manifest.jsonl"),
            (Join-Path $inputRoot "frames")
        )) {
            if (-not (Test-Path -LiteralPath $required)) {
                throw "Required input is missing: $required"
            }
        }
        Assert-RemoteAbsent -Path $remoteRun
        Invoke-Adb -Arguments @("shell", "mkdir", "-p", "$remoteRun/input")
        Invoke-Adb -Arguments @(
            "push",
            (Join-Path $inputRoot "input_receipt.json"),
            "$remoteRun/input/"
        )
        Invoke-Adb -Arguments @(
            "push",
            (Join-Path $inputRoot "manifest.jsonl"),
            "$remoteRun/input/"
        )
        Invoke-Adb -Arguments @(
            "push",
            (Join-Path $inputRoot "frames"),
            "$remoteRun/input/"
        )
    }
    "RunBaseline" {
        Assert-RemoteAbsent -Path "$remoteRun/baseline-output"
        Assert-RemoteAbsent -Path "$remoteRun/candidate-output"
        Assert-RemoteAbsent -Path "$remoteRun/candidate_authorization.json"
        Invoke-Instrumentation -Method "runUnseenNaturalRank2BaselineOnly"
    }
    "CollectBaseline" {
        Collect-RemoteOutput `
            -Remote "$remoteRun/baseline-output" `
            -Local $baselineRoot
    }
    "EvaluateBaseline" {
        Push-Location $repoRoot
        try {
            & python -m `
                scripts.research.dual_loop_unseen_natural_event_r0.evaluate_rank2_effect `
                baseline `
                --baseline-root $baselineRoot `
                --truth-root $truthRoot `
                --input-root $inputRoot `
                --build-identity $buildIdentity `
                --output $baselineEvaluationRoot
            if ($LASTEXITCODE -ne 0) {
                throw "Baseline adequacy evaluation failed"
            }
        } finally {
            Pop-Location
        }
    }
    "StageAuthorization" {
        if (-not (Test-Path -LiteralPath $authorization)) {
            throw "Candidate authorization is missing: $authorization"
        }
        Push-Location $repoRoot
        try {
            & python -m `
                scripts.research.dual_loop_unseen_natural_event_r0.evaluate_rank2_effect `
                verify-authorization `
                --baseline-root $baselineRoot `
                --truth-root $truthRoot `
                --input-root $inputRoot `
                --build-identity $buildIdentity `
                --authorization $authorization
            if ($LASTEXITCODE -ne 0) {
                throw "Candidate authorization did not survive baseline recomputation"
            }
        } finally {
            Pop-Location
        }
        $value = Get-Content -LiteralPath $authorization -Raw -Encoding UTF8 |
            ConvertFrom-Json
        if (
            $value.status -ne "AUTHORIZED" -or
            -not $value.baseline_adequacy -or
            $value.candidate_output_opened
        ) {
            throw "Candidate authorization is invalid"
        }
        if (-not (Test-Path -LiteralPath $assessment)) {
            throw "Baseline assessment is missing: $assessment"
        }
        $assessmentValue =
            Get-Content -LiteralPath $assessment -Raw -Encoding UTF8 |
                ConvertFrom-Json
        $evaluatorPath = Join-Path $PSScriptRoot "evaluate_rank2_effect.py"
        $evaluatorSha = (
            Get-FileHash -LiteralPath $evaluatorPath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        if (
            $assessmentValue.status -ne "BASELINE_ADEQUATE" -or
            -not $assessmentValue.candidate_authorized -or
            $assessmentValue.baseline_hit_positive_count -lt 1 -or
            $assessmentValue.baseline_alerted_negative_count -lt 1 -or
            $value.evaluator_implementation_sha256 -ne $evaluatorSha -or
            $value.baseline_assessment_sha256 -ne (
                Get-FileHash -LiteralPath $assessment -Algorithm SHA256
            ).Hash.ToLowerInvariant()
        ) {
            throw "Candidate authorization/assessment binding is invalid"
        }
        if (
            (Get-InstalledApkSha256 -PackageName "com.linnan.blindassist") -ne
                $value.installed_app_apk_sha256 -or
            (
                Get-InstalledApkSha256 `
                    -PackageName "com.linnan.blindassist.benchmark"
            ) -ne $value.installed_test_apk_sha256
        ) {
            throw "Installed APK identity changed after baseline"
        }
        Assert-RemoteAbsent -Path "$remoteRun/candidate-output"
        Assert-RemoteAbsent -Path "$remoteRun/candidate_authorization.json"
        Assert-RemoteAbsent -Path "$remoteRun/baseline_assessment.json"
        Invoke-Adb -Arguments @(
            "push",
            $assessment,
            "$remoteRun/baseline_assessment.json"
        )
        Invoke-Adb -Arguments @(
            "push",
            $authorization,
            "$remoteRun/candidate_authorization.json"
        )
    }
    "RunCandidate" {
        Assert-RemoteAbsent -Path "$remoteRun/candidate-output"
        Invoke-Instrumentation -Method "runUnseenNaturalRank2CandidateOnly"
    }
    "CollectCandidate" {
        Collect-RemoteOutput `
            -Remote "$remoteRun/candidate-output" `
            -Local $candidateRoot
    }
    "EvaluateCandidate" {
        Push-Location $repoRoot
        try {
            & python -m `
                scripts.research.dual_loop_unseen_natural_event_r0.evaluate_rank2_effect `
                candidate `
                --baseline-root $baselineRoot `
                --candidate-root $candidateRoot `
                --truth-root $truthRoot `
                --input-root $inputRoot `
                --build-identity $buildIdentity `
                --authorization $authorization `
                --output $effectEvaluationRoot
            if ($LASTEXITCODE -ne 0) {
                throw "Candidate effect evaluation failed"
            }
        } finally {
            Pop-Location
        }
    }
}
