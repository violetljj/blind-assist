[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Build", "Install", "PrepareInput", "Prestart", "StartFormal", "CollectFormal")]
    [string]$Action,
    [string]$DeviceSerial = "R5CX10M8Y8X",
    [string]$ActivationReceipt =
        "artifacts.local/evidence/dual-loop/production-temporal-geometry-factorial-ab-r0/implementation/activation.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$adb = (Get-Command adb -ErrorAction Stop).Source
$package = "com.linnan.blindassist"
$instrumentation = "com.linnan.blindassist.benchmark/androidx.test.runner.AndroidJUnitRunner"
$class = "com.linnan.blindassist.benchmark.ProductionTemporalGeometryFactorialAbDeviceTest"
$protocolId = "DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0"
$implementationId = "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0"
$remoteBase = "/sdcard/Android/data/$package/files/dual_loop_production_temporal_ab_r0"
$hostEvidence = Join-Path $repoRoot "artifacts.local/evidence/dual-loop/production-temporal-geometry-factorial-ab-r0"
$appApk = Join-Path $repoRoot "app/build/outputs/apk/debug/app-debug.apk"
$testApk = Join-Path $repoRoot "apps/benchmarks/device-benchmark/build/outputs/apk/debug/device-benchmark-debug.apk"

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
    $release = (& $adb -s $DeviceSerial shell getprop ro.build.version.release).Trim()
    $sdk = (& $adb -s $DeviceSerial shell getprop ro.build.version.sdk).Trim()
    $abis = (& $adb -s $DeviceSerial shell getprop ro.product.cpu.abilist).Trim()
    if (
        $model -ne "SM-S9280" -or
        $soc -ne "SM8650" -or
        $release -ne "16" -or
        $sdk -ne "36" -or
        $abis -ne "arm64-v8a"
    ) {
        throw "Device identity mismatch: model=$model soc=$soc release=$release sdk=$sdk abis=$abis"
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

Assert-Device

switch ($Action) {
    "Build" {
        $env:JAVA_HOME = "E:\codex-tools\jdk-17"
        if (-not (Test-Path -LiteralPath $env:JAVA_HOME)) {
            $env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
        }
        & (Join-Path $repoRoot "gradlew.bat") :app:assembleDebug :device-benchmark:assembleDebug --no-daemon
        if ($LASTEXITCODE -ne 0) { throw "Gradle Android build failed" }
    }
    "Install" {
        if (-not (Test-Path -LiteralPath $appApk) -or -not (Test-Path -LiteralPath $testApk)) {
            throw "Build APKs before installation"
        }
        Invoke-Adb -Arguments @("install", "-r", $appApk)
        Invoke-Adb -Arguments @("install", "-r", $testApk)
        $installed = & $adb -s $DeviceSerial shell pm list instrumentation
        if (($installed -join "`n") -notmatch [regex]::Escape($instrumentation)) {
            throw "Expected instrumentation is not installed"
        }
    }
    "PrepareInput" {
        $sourceRoot = Join-Path $repoRoot "artifacts.local/camera-source-prescreen-r1/dataset/crowdbot_0327_shared_control/sequences"
        $remoteInput = "$remoteBase/input"
        $existing = (& $adb -s $DeviceSerial shell "if [ -e '$remoteInput' ]; then echo EXISTS; else echo ABSENT; fi").Trim()
        if ($existing -ne "ABSENT") {
            throw "Remote input already exists; refusing to overwrite: $remoteInput"
        }
        Invoke-Adb -Arguments @("shell", "mkdir", "-p", $remoteInput)
        foreach ($session in @(
            "defaced_2021-03-27-11-51-18_filtered_lidar_odom",
            "defaced_2021-03-27-11-55-00_filtered_lidar_odom"
        )) {
            $localSession = Join-Path $sourceRoot $session
            $remoteSession = "$remoteInput/$session"
            Invoke-Adb -Arguments @("shell", "mkdir", "-p", $remoteSession)
            Invoke-Adb -Arguments @("push", (Join-Path $localSession "frames.jsonl"), "$remoteSession/")
            Invoke-Adb -Arguments @("push", (Join-Path $localSession "rgb"), "$remoteSession/")
        }
    }
    "Prestart" {
        $instrumentationOutput = & $adb -s $DeviceSerial `
            shell am instrument -w -r -e class `
            "$class#verifyFormalInputAndQnnPrestart" $instrumentation
        $instrumentationText = $instrumentationOutput -join "`n"
        $instrumentationOutput | Write-Output
        if ($LASTEXITCODE -ne 0 -or $instrumentationText -notmatch "OK \(1 test\)") {
            throw "Prestart instrumentation did not complete successfully"
        }
        $hostPrestart = Join-Path $hostEvidence "implementation/prestart_receipt.json"
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $hostPrestart) | Out-Null
        Invoke-Adb -Arguments @("pull", "$remoteBase/prestart/prestart_receipt.json", $hostPrestart)
    }
    "StartFormal" {
        $activationPath = Join-Path $repoRoot $ActivationReceipt
        if (-not (Test-Path -LiteralPath $activationPath)) {
            throw "Activation receipt is missing: $activationPath"
        }
        $activation = Get-Content -LiteralPath $activationPath -Raw -Encoding utf8 | ConvertFrom-Json
        if (
            $activation.schema_version -ne "blindassist.production_temporal_ab_activation.v1" -or
            $activation.protocol_id -ne $protocolId -or
            $activation.implementation_id -ne $implementationId -or
            $activation.status -ne "ACTIVATED" -or
            -not $activation.formal_execution_authorized -or
            $activation.device_serial -ne $DeviceSerial
        ) {
            throw "Activation receipt does not authorize formal execution"
        }
        $gitStatus = & git -C $repoRoot status --short
        if ($LASTEXITCODE -ne 0 -or ($gitStatus -join "`n").Trim()) {
            throw "Formal launch requires a clean worktree"
        }
        $head = (& git -C $repoRoot rev-parse HEAD).Trim()
        $originMaster = (& git -C $repoRoot rev-parse origin/master).Trim()
        if ($LASTEXITCODE -ne 0 -or $head -ne $originMaster -or $head -ne $activation.git_commit) {
            throw "Formal launch Git identity drift"
        }
        $bindings = @(
            [pscustomobject]@{
                Path = $activation.implementation_lock_path
                Hash = $activation.implementation_lock_sha256
                Label = "implementation lock"
            },
            [pscustomobject]@{
                Path = $activation.implementation_review_path
                Hash = $activation.implementation_review_sha256
                Label = "implementation review"
            }
        )
        foreach ($binding in $bindings) {
            if (-not (Test-Path -LiteralPath $binding.Path)) {
                throw "Formal launch binding is missing: $($binding.Label)"
            }
            $actualHash = (Get-FileHash -LiteralPath $binding.Path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualHash -ne $binding.Hash) {
                throw "Formal launch binding drift: $($binding.Label)"
            }
        }
        if (
            (Get-InstalledApkSha256 -PackageName "com.linnan.blindassist") -ne
                $activation.installed_app_apk_sha256 -or
            (Get-InstalledApkSha256 -PackageName "com.linnan.blindassist.benchmark") -ne
                $activation.installed_test_apk_sha256
        ) {
            throw "Formal launch installed APK identity drift"
        }
        foreach ($relative in @("device-producer", "sealed-producer", "evaluation")) {
            if (Test-Path -LiteralPath (Join-Path $hostEvidence $relative)) {
                throw "Host candidate namespace already exists: $relative"
            }
        }
        $remoteState = (& $adb -s $DeviceSerial shell "if [ -e '$remoteBase/formal_start.json' ] || [ -e '$remoteBase/output' ] || [ -e '$remoteBase/authorization' ]; then echo EXISTS; else echo ABSENT; fi").Trim()
        if ($remoteState -ne "ABSENT") {
            throw "Remote formal marker/output already exists; one-shot cannot start"
        }
        $runRoot = Join-Path $hostEvidence "formal-process"
        if (Test-Path -LiteralPath $runRoot) {
            throw "Formal launch lock already exists: $runRoot"
        }
        New-Item -ItemType Directory -Path $runRoot | Out-Null
        Invoke-Adb -Arguments @("shell", "mkdir", "$remoteBase/authorization")
        Invoke-Adb -Arguments @(
            "push",
            $activationPath,
            "$remoteBase/authorization/activation.json"
        )
        Invoke-Adb -Arguments @(
            "push",
            $activation.implementation_lock_path,
            "$remoteBase/authorization/implementation_lock.json"
        )
        $stdout = Join-Path $runRoot "instrumentation.stdout.log"
        $stderr = Join-Path $runRoot "instrumentation.stderr.log"
        $arguments = @(
            "-s", $DeviceSerial, "shell", "am", "instrument", "-w", "-r",
            "-e", "class", "$class#runFormalTruthBlindProducer", $instrumentation
        )
        $process = Start-Process -FilePath $adb -ArgumentList $arguments -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $processReceipt = [ordered]@{
            status = "RUNNING"
            pid = $process.Id
            device_serial = $DeviceSerial
            stdout = $stdout
            stderr = $stderr
        } | ConvertTo-Json
        Set-Content -LiteralPath (Join-Path $runRoot "process.json") -Value $processReceipt -Encoding utf8
        Write-Output "Formal instrumentation started with PID $($process.Id)"
    }
    "CollectFormal" {
        $processReceiptPath = Join-Path $hostEvidence "formal-process/process.json"
        if (-not (Test-Path -LiteralPath $processReceiptPath)) {
            throw "Formal process receipt is missing"
        }
        $processReceipt = Get-Content -LiteralPath $processReceiptPath -Raw -Encoding utf8 | ConvertFrom-Json
        if (Get-Process -Id $processReceipt.pid -ErrorAction SilentlyContinue) {
            throw "Formal instrumentation is still running: PID $($processReceipt.pid)"
        }
        $destination = Join-Path $hostEvidence "device-producer"
        if (Test-Path -LiteralPath $destination) {
            throw "Host device-producer already exists"
        }
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        Invoke-Adb -Arguments @("pull", "$remoteBase/output/device-producer/.", $destination)
        Invoke-Adb -Arguments @("pull", "$remoteBase/formal_start.json", (Join-Path $destination "formal_start.json"))
    }
}
