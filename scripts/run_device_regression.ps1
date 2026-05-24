param(
    [string]$ApkPath = "app\build\outputs\apk\debug\app-debug.apk",
    [string]$PackageName = "com.linnan.blindassist",
    [string]$MainActivity = ".MainActivity",
    [int]$SampleSeconds = 90,
    [switch]$RunConnectedAndroidTest,
    [string]$AdbPath
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $PSScriptRoot "..\$Path")
}

function Resolve-Adb([string]$RequestedPath) {
    if ($RequestedPath) {
        if (-not (Test-Path -LiteralPath $RequestedPath)) {
            throw "ADB not found at $RequestedPath"
        }
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    $localAdb = Join-Path $PSScriptRoot "..\.android-sdk\platform-tools\adb.exe"
    if (Test-Path -LiteralPath $localAdb) {
        return (Resolve-Path -LiteralPath $localAdb).Path
    }

    $command = Get-Command adb -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "ADB not found. Pass -AdbPath or install platform-tools."
}

function Invoke-Adb {
    param(
        [string]$Adb,
        [string[]]$Arguments,
        [string]$OutFile
    )

    $text = & $Adb @Arguments 2>&1
    $code = $LASTEXITCODE
    if ($OutFile) {
        $text | Out-File -FilePath $OutFile -Encoding utf8
    } else {
        $text
    }
    if ($code -ne 0) {
        throw "adb $($Arguments -join ' ') failed with exit code $code"
    }
    return $text
}

function Get-SingleDevice([string]$Adb) {
    $lines = & $Adb devices 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "adb devices failed: $($lines -join ' ')"
    }

    $devices = @(
        $lines |
            Where-Object { $_ -match "^\S+\s+device$" } |
            ForEach-Object { ($_ -split "\s+")[0] }
    )

    if ($devices.Count -ne 1) {
        throw "Expected exactly one online device, found $($devices.Count). Raw adb devices output: $($lines -join ' | ')"
    }
    return $devices[0]
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$resolvedApk = Resolve-RepoPath $ApkPath
if (-not (Test-Path -LiteralPath $resolvedApk)) {
    throw "APK not found: $resolvedApk"
}
$resolvedApk = (Resolve-Path -LiteralPath $resolvedApk).Path
$adb = Resolve-Adb $AdbPath
$device = Get-SingleDevice $adb

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = Join-Path $repoRoot "test-artifacts.local-device-regression-$timestamp"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$summary = [ordered]@{
    timestamp = $timestamp
    device = $device
    apk = $resolvedApk
    packageName = $PackageName
    mainActivity = $MainActivity
    sampleSeconds = $SampleSeconds
    runConnectedAndroidTest = [bool]$RunConnectedAndroidTest
    artifactRoot = $artifactRoot
}

try {
    Invoke-Adb $adb @("devices", "-l") (Join-Path $artifactRoot "adb-devices.txt") | Out-Null
    Invoke-Adb $adb @("-s", $device, "shell", "getprop") (Join-Path $artifactRoot "device-getprop.txt") | Out-Null

    Invoke-Adb $adb @("-s", $device, "install", "-r", "-t", $resolvedApk) (Join-Path $artifactRoot "install.txt") | Out-Null
    Invoke-Adb $adb @("-s", $device, "shell", "pm", "clear", $PackageName) (Join-Path $artifactRoot "pm-clear.txt") | Out-Null

    $component = if ($MainActivity.StartsWith(".")) { "$PackageName/$MainActivity" } else { "$PackageName/$MainActivity" }
    Invoke-Adb $adb @("-s", $device, "shell", "am", "start", "-W", "-n", $component) (Join-Path $artifactRoot "cold-start.txt") | Out-Null

    Start-Sleep -Seconds 3
    Invoke-Adb $adb @("-s", $device, "shell", "dumpsys", "package", $PackageName) (Join-Path $artifactRoot "dumpsys-package.txt") | Out-Null
    Invoke-Adb $adb @("-s", $device, "shell", "cmd", "package", "dump", $PackageName) (Join-Path $artifactRoot "cmd-package-dump.txt") | Out-Null

    if ($RunConnectedAndroidTest) {
        Push-Location $repoRoot
        try {
            .\gradlew.bat connectedDebugAndroidTest --no-daemon 2>&1 |
                Tee-Object -FilePath (Join-Path $artifactRoot "connectedDebugAndroidTest.txt")
            if ($LASTEXITCODE -ne 0) {
                throw "connectedDebugAndroidTest failed with exit code $LASTEXITCODE"
            }
        } finally {
            Pop-Location
        }
    }

    $stopAt = (Get-Date).AddSeconds($SampleSeconds)
    $iteration = 0
    while ((Get-Date) -lt $stopAt) {
        $iteration++
        $suffix = "{0:D3}" -f $iteration
        Invoke-Adb $adb @("-s", $device, "logcat", "-d", "-s", "BlindAssistPerf") (Join-Path $artifactRoot "BlindAssistPerf-$suffix.log") | Out-Null
        Invoke-Adb $adb @("-s", $device, "shell", "dumpsys", "gfxinfo", $PackageName) (Join-Path $artifactRoot "gfxinfo-$suffix.txt") | Out-Null
        Invoke-Adb $adb @("-s", $device, "shell", "dumpsys", "meminfo", $PackageName) (Join-Path $artifactRoot "meminfo-$suffix.txt") | Out-Null
        Invoke-Adb $adb @("-s", $device, "exec-out", "uiautomator", "dump", "/dev/tty") (Join-Path $artifactRoot "ui-dump-$suffix.xml") | Out-Null
        & $adb -s $device exec-out screencap -p > (Join-Path $artifactRoot "screenshot-$suffix.png")
        if ($LASTEXITCODE -ne 0) {
            throw "screenshot capture failed with exit code $LASTEXITCODE"
        }
        Start-Sleep -Seconds 15
    }

    $summary.status = "passed"
} catch {
    $summary.status = "failed"
    $summary.error = $_.Exception.Message
    throw
} finally {
    $summary | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $artifactRoot "summary.json") -Encoding utf8
    "Device regression artifacts: $artifactRoot" | Out-File -FilePath (Join-Path $artifactRoot "README.txt") -Encoding utf8
    Write-Host "Device regression artifacts: $artifactRoot"
}
