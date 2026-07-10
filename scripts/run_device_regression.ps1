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

function Invoke-NativeAdb {
    param(
        [string]$Adb,
        [string[]]$Arguments
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $text = & $Adb @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }

    return [ordered]@{
        Text = $text
        Code = $code
    }
}

function Invoke-Adb {
    param(
        [string]$Adb,
        [string[]]$Arguments,
        [string]$OutFile
    )

    $result = Invoke-NativeAdb $Adb $Arguments
    $text = $result.Text
    $code = $result.Code
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
    $result = Invoke-NativeAdb $Adb @("devices")
    $lines = $result.Text
    if ($result.Code -ne 0) {
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

function Get-UiHierarchy([string]$Adb, [string]$Device) {
    $dump = Invoke-NativeAdb $Adb @("-s", $Device, "shell", "uiautomator", "dump", "/sdcard/blindassist-regression-ui.xml")
    if ($dump.Code -ne 0) {
        throw "uiautomator dump failed: $($dump.Text -join ' ')"
    }

    $content = Invoke-NativeAdb $Adb @("-s", $Device, "shell", "cat", "/sdcard/blindassist-regression-ui.xml")
    if ($content.Code -ne 0) {
        throw "reading uiautomator hierarchy failed: $($content.Text -join ' ')"
    }
    return [xml](($content.Text) -join "")
}

function Find-UiNodeByText([xml]$Hierarchy, [string[]]$Texts) {
    foreach ($text in $Texts) {
        $node = $Hierarchy.SelectNodes("//node") |
            Where-Object {
                $_.GetAttribute("text") -eq $text -or
                $_.GetAttribute("content-desc") -eq $text
            } |
            Select-Object -First 1
        if ($node) {
            return $node
        }
    }
    return $null
}

function Invoke-UiTapByText {
    param(
        [string]$Adb,
        [string]$Device,
        [string[]]$Texts,
        [switch]$Optional
    )

    $hierarchy = Get-UiHierarchy $Adb $Device
    $node = Find-UiNodeByText $hierarchy $Texts
    if (-not $node) {
        if ($Optional) {
            return $false
        }
        throw "None of the expected UI texts were found: $($Texts -join ' | ')"
    }

    while ($node -and $node.GetAttribute("clickable") -ne "true") {
        $node = $node.ParentNode
    }
    if (-not $node) {
        throw "Expected UI text was found, but no clickable ancestor exists: $($Texts -join ' | ')"
    }

    $bounds = $node.GetAttribute("bounds")
    if ($bounds -notmatch '^\[(\d+),(\d+)\]\[(\d+),(\d+)\]$') {
        throw "Unable to parse clickable UI bounds: $bounds"
    }
    $left = [int]$Matches[1]
    $top = [int]$Matches[2]
    $right = [int]$Matches[3]
    $bottom = [int]$Matches[4]
    $centerX = [int](($left + $right) / 2)
    $centerY = [int](($top + $bottom) / 2)
    Invoke-Adb $Adb @("-s", $Device, "shell", "input", "tap", "$centerX", "$centerY") $null | Out-Null
    return $true
}

function Wait-ForUiText {
    param(
        [string]$Adb,
        [string]$Device,
        [string[]]$Texts,
        [int]$TimeoutSeconds = 15
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $hierarchy = Get-UiHierarchy $Adb $Device
        if (Find-UiNodeByText $hierarchy $Texts) {
            return $hierarchy
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)

    throw "Timed out waiting for UI text: $($Texts -join ' | ')"
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
$artifactRoot = Join-Path (Join-Path $repoRoot "test-artifacts.local\device-regression") $timestamp
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

    if ($RunConnectedAndroidTest) {
        Push-Location $repoRoot
        try {
            .\gradlew.bat :app:connectedDebugAndroidTest --no-daemon 2>&1 |
                Tee-Object -FilePath (Join-Path $artifactRoot "connectedDebugAndroidTest.txt")
            if ($LASTEXITCODE -ne 0) {
                throw ":app:connectedDebugAndroidTest failed with exit code $LASTEXITCODE"
            }
        } finally {
            Pop-Location
        }
    }

    Invoke-Adb $adb @("-s", $device, "install", "-r", "-t", $resolvedApk) (Join-Path $artifactRoot "install.txt") | Out-Null
    Invoke-Adb $adb @("-s", $device, "shell", "pm", "clear", $PackageName) (Join-Path $artifactRoot "pm-clear.txt") | Out-Null
    Invoke-Adb $adb @("-s", $device, "shell", "pm", "grant", $PackageName, "android.permission.CAMERA") (Join-Path $artifactRoot "pm-grant-camera.txt") | Out-Null
    Invoke-Adb $adb @("-s", $device, "logcat", "-c") $null | Out-Null

    $component = if ($MainActivity.StartsWith(".")) { "$PackageName/$MainActivity" } else { "$PackageName/$MainActivity" }
    Invoke-Adb $adb @("-s", $device, "shell", "am", "start", "-W", "-n", $component) (Join-Path $artifactRoot "cold-start.txt") | Out-Null

    Start-Sleep -Seconds 2
    if (Invoke-UiTapByText -Adb $adb -Device $device -Texts @("不再显示", "Don't show again", "Do not show again") -Optional) {
        Start-Sleep -Seconds 1
    } elseif (Invoke-UiTapByText -Adb $adb -Device $device -Texts @("确定", "OK") -Optional) {
        Start-Sleep -Seconds 1
    }
    if (Invoke-UiTapByText -Adb $adb -Device $device -Texts @("跳过引导", "Skip guide", "Skip onboarding") -Optional) {
        Start-Sleep -Seconds 2
    }
    Invoke-UiTapByText -Adb $adb -Device $device -Texts @("使用手机摄像头", "Use phone camera") | Out-Null
    $cameraHierarchy = Wait-ForUiText -Adb $adb -Device $device -Texts @("检测中", "Detecting") -TimeoutSeconds 20
    $cameraHierarchy.OuterXml | Out-File -FilePath (Join-Path $artifactRoot "camera-ready-ui.xml") -Encoding utf8
    Start-Sleep -Seconds 5

    $foreground = Invoke-NativeAdb $adb @("-s", $device, "shell", "dumpsys", "activity", "activities")
    $foreground.Text | Out-File -FilePath (Join-Path $artifactRoot "foreground-activity.txt") -Encoding utf8
    if (($foreground.Text -join "`n") -notmatch "(topResumedActivity|ResumedActivity|mResumedActivity).*$([regex]::Escape($PackageName))") {
        throw "BlindAssist is not the resumed foreground activity after camera preparation"
    }

    $initialPerf = Invoke-NativeAdb $adb @("-s", $device, "logcat", "-d", "-s", "BlindAssistPerf")
    $initialPerf.Text | Out-File -FilePath (Join-Path $artifactRoot "BlindAssistPerf-initial.log") -Encoding utf8
    $initialPerfText = $initialPerf.Text -join "`n"
    $initialPerfLineCount = @($initialPerf.Text | Where-Object { $_ -match "BlindAssistPerf:" }).Count
    if ($initialPerfText -notmatch "status=(模型已加载|model loaded)") {
        throw "No model-ready BlindAssistPerf frame was observed after entering the camera"
    }

    Invoke-Adb $adb @("-s", $device, "shell", "dumpsys", "package", $PackageName) (Join-Path $artifactRoot "dumpsys-package.txt") | Out-Null
    Invoke-Adb $adb @("-s", $device, "shell", "cmd", "package", "dump", $PackageName) (Join-Path $artifactRoot "cmd-package-dump.txt") | Out-Null

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

    $finalHierarchy = Wait-ForUiText -Adb $adb -Device $device -Texts @("检测中", "Detecting") -TimeoutSeconds 5
    $finalHierarchy.OuterXml | Out-File -FilePath (Join-Path $artifactRoot "camera-final-ui.xml") -Encoding utf8

    $finalForeground = Invoke-NativeAdb $adb @("-s", $device, "shell", "dumpsys", "activity", "activities")
    $finalForeground.Text | Out-File -FilePath (Join-Path $artifactRoot "foreground-activity-final.txt") -Encoding utf8
    if (($finalForeground.Text -join "`n") -notmatch "(topResumedActivity|ResumedActivity|mResumedActivity).*$([regex]::Escape($PackageName))") {
        throw "BlindAssist is not the resumed foreground activity after the sampling interval"
    }

    $finalPerf = Invoke-NativeAdb $adb @("-s", $device, "logcat", "-d", "-s", "BlindAssistPerf")
    $finalPerf.Text | Out-File -FilePath (Join-Path $artifactRoot "BlindAssistPerf-final.log") -Encoding utf8
    $finalPerfLineCount = @($finalPerf.Text | Where-Object { $_ -match "BlindAssistPerf:" }).Count
    if ($finalPerfLineCount -le $initialPerfLineCount) {
        throw "No new BlindAssistPerf frames were observed during the sampling interval"
    }

    $finalLogcat = Invoke-NativeAdb $adb @("-s", $device, "logcat", "-d", "-v", "time")
    $finalLogcat.Text | Out-File -FilePath (Join-Path $artifactRoot "logcat-final.txt") -Encoding utf8
    $finalLogText = $finalLogcat.Text -join "`n"
    $escapedPackage = [regex]::Escape($PackageName)
    $targetCrash = $finalLogText -match "(?s)FATAL EXCEPTION.{0,1200}Process:\s*$escapedPackage" -or
        $finalLogText -match "ANR in $escapedPackage" -or
        $finalLogText -match "am_crash.*$escapedPackage" -or
        $finalLogText -match "Process $escapedPackage .* has died" -or
        $finalLogText -match "(?s)Fatal signal.{0,1200}>>>\s*$escapedPackage\s*<<<"
    if ($targetCrash) {
        throw "Crash or ANR evidence was found in the final device log"
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
