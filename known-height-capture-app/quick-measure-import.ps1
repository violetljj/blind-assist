param(
    [ValidateSet('Near', 'Middle', 'Far')]
    [string]$Slot = 'Near',
    [string]$Serial = 'R5CX10M8Y8X',
    [int]$TimeoutSeconds = 90,
    [switch]$LaunchQuickMeasure,
    [string]$ExistingScreenshot
)

$ErrorActionPreference = 'Stop'
$adb = 'E:\codex-tools\bin\adb.cmd'
$capturePackage = 'com.linnan.blindassist.heightcapture'
$captureActivity = 'com.linnan.blindassist.ustrfbenchmark.KnownHeightCaptureActivity'
$evidenceRoot = 'E:\linnan\linnan\artifacts.local\quick-measure-captures'

function Invoke-Adb([string[]]$Arguments) {
    & $adb -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "ADB failed: $($Arguments -join ' ')"
    }
}

function Get-FocusedWindow {
    $line = Invoke-Adb @('shell', 'dumpsys', 'window') |
        Select-String 'mCurrentFocus=' |
        Select-Object -First 1
    return [string]$line
}

function Initialize-WindowsOcr {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
    [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime] | Out-Null
    [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
    [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
    [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null

    $script:AsTaskMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 } |
        Select-Object -First 1
    $script:OcrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage(
        [Windows.Globalization.Language]::new('zh-Hans')
    )
    if ($null -eq $script:OcrEngine) {
        throw 'Windows Chinese OCR is unavailable.'
    }
}

function Wait-WinRt($Operation, [Type]$ResultType) {
    $task = $script:AsTaskMethod.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

function Read-OcrText([string]$ImagePath) {
    $file = Wait-WinRt ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
    $stream = Wait-WinRt ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder = Wait-WinRt ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Wait-WinRt ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        try {
            $result = Wait-WinRt ($script:OcrEngine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
            return $result.Text
        } finally {
            $bitmap.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Find-DistanceMeters([string]$Text) {
    $normalized = $Text -replace '\s+', ' '
    $patterns = @(
        @{ Regex = '(?<value>\d+(?:[\.,]\d+)?)\s*(?:\u5398\s*\u7c73|cm)'; Factor = 0.01 },
        @{ Regex = '(?<value>\d+(?:[\.,]\d+)?)\s*\u7c73'; Factor = 1.0 }
    )
    foreach ($pattern in $patterns) {
        $match = [regex]::Match($normalized, $pattern.Regex, 'IgnoreCase')
        if ($match.Success) {
            $value = [double]::Parse($match.Groups['value'].Value.Replace(',', '.'), [Globalization.CultureInfo]::InvariantCulture)
            return $value * $pattern.Factor
        }
    }
    return $null
}

function Set-DistanceField([double]$Meters) {
    Invoke-Adb @('shell', 'am', 'start', '--user', '0', '-n', "$capturePackage/$captureActivity") | Out-Null
    Start-Sleep -Milliseconds 900
    if ((Get-FocusedWindow) -notmatch [regex]::Escape($capturePackage)) {
        throw 'Height capture app did not become the focused window.'
    }

    Invoke-Adb @('shell', 'input', 'keyevent', '4') | Out-Null
    Start-Sleep -Milliseconds 200
    $deviceHierarchy = '/sdcard/quick_measure_import_window.xml'
    Invoke-Adb @('shell', 'uiautomator', 'dump', $deviceHierarchy) | Out-Null
    $hierarchy = (Invoke-Adb @('shell', 'cat', $deviceHierarchy) | Out-String)
    $fields = [regex]::Matches(
        $hierarchy,
        '<node[^>]*class="android.widget.EditText"[^>]*bounds="\[(?<x1>\d+),(?<y1>\d+)\]\[(?<x2>\d+),(?<y2>\d+)\]"'
    )
    if ($fields.Count -ge 3) {
        $field = $fields[$fields.Count - 1]
        $x = ([int]$field.Groups['x1'].Value + [int]$field.Groups['x2'].Value) / 2
        $y = ([int]$field.Groups['y1'].Value + [int]$field.Groups['y2'].Value) / 2
    } else {
        $x = 720
        $y = 2179
    }
    $formatted = ($Meters * 100.0).ToString('0.##', [Globalization.CultureInfo]::InvariantCulture)

    Invoke-Adb @('shell', 'input', 'tap', "$x", "$y") | Out-Null
    Start-Sleep -Milliseconds 200
    Invoke-Adb @('shell', 'input', 'keyevent', '123') | Out-Null
    1..12 | ForEach-Object { Invoke-Adb @('shell', 'input', 'keyevent', '67') | Out-Null }
    Invoke-Adb @('shell', 'input', 'text', $formatted) | Out-Null
    Invoke-Adb @('shell', 'input', 'keyevent', '4') | Out-Null
    return $Meters.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)
}

New-Item -ItemType Directory -Force -Path $evidenceRoot | Out-Null
Initialize-WindowsOcr

if ($LaunchQuickMeasure) {
    Invoke-Adb @('shell', 'am', 'start', '-a', 'android.intent.action.VIEW', '-d', 'ruler://com.samsung.android.ruler') | Out-Null
}

$stableValue = $null
$acceptedImage = $null
$stableCount = 0
if ($ExistingScreenshot) {
    $acceptedImage = (Resolve-Path -LiteralPath $ExistingScreenshot).Path
    $existingOcrText = Read-OcrText $acceptedImage
    $stableValue = Find-DistanceMeters $existingOcrText
    if ($null -eq $stableValue) {
        throw "No Quick Measure distance was recognized. OCR text: $existingOcrText"
    }
} else {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ((Get-FocusedWindow) -notmatch 'com\.samsung\.android\.ruler') {
            Start-Sleep -Milliseconds 400
            continue
        }
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        $candidate = Join-Path $evidenceRoot "quick-measure-$stamp.png"
        & $adb -s $Serial exec-out screencap -p > $candidate
        if ($LASTEXITCODE -ne 0) { throw 'Unable to capture Quick Measure screen.' }
        $meters = Find-DistanceMeters (Read-OcrText $candidate)
        if ($null -eq $meters) {
            Remove-Item -LiteralPath $candidate
            Start-Sleep -Milliseconds 350
            continue
        }
        if ($null -ne $stableValue -and [Math]::Abs($meters - $stableValue) -lt 0.0001) {
            $stableCount++
        } else {
            $stableValue = $meters
            $stableCount = 1
        }
        if ($stableCount -ge 2) {
            $acceptedImage = $candidate
            break
        }
        Remove-Item -LiteralPath $candidate
        Start-Sleep -Milliseconds 350
    }
}

if ($null -eq $acceptedImage) {
    throw "No stable Quick Measure distance was recognized within $TimeoutSeconds seconds."
}

$written = Set-DistanceField $stableValue
$receipt = [IO.Path]::ChangeExtension($acceptedImage, '.txt')
@(
    "distance_m=$written"
    "slot=$Slot"
    'source=Samsung Quick Measure screen OCR'
    "captured_at=$([DateTimeOffset]::Now.ToString('o'))"
    "screenshot=$acceptedImage"
) | Set-Content -LiteralPath $receipt -Encoding utf8

[pscustomobject]@{
    ok = $true
    distance_m = $written
    slot = $Slot
    screenshot = $acceptedImage
    receipt = $receipt
    app = $capturePackage
} | ConvertTo-Json -Compress
