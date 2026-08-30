[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$PreflightOnly,
    [switch]$RequireDevice,
    [string]$AndroidSerial,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GradleArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Stop-Environment {
    param([Parameter(Mandatory = $true)][string]$Message)
    [Console]::Error.WriteLine("ENV_BLOCKED: $Message")
    exit 20
}

function Stop-Usage {
    param([Parameter(Mandatory = $true)][string]$Message)
    [Console]::Error.WriteLine("USAGE_ERROR: $Message")
    exit 64
}

function Get-DeclaredVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Content,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $pattern = '(?m)^\s*' + [regex]::Escape($Name) +
        '\s*=\s*"(?<version>[^"]+)"\s*$'
    $match = [regex]::Match($Content, $pattern)
    if (-not $match.Success) {
        Stop-Environment "Version catalog does not declare '$Name'."
    }
    return $match.Groups["version"].Value
}

function Get-JavaVersion {
    param([Parameter(Mandatory = $true)][string]$JavaHome)
    $java = Join-Path $JavaHome "bin\java.exe"
    if (-not (Test-Path -LiteralPath $java -PathType Leaf)) {
        return $null
    }
    $output = @(& $java -version 2>&1)
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    $match = [regex]::Match(
        ($output -join [Environment]::NewLine),
        '(?m)(?:openjdk|java) version "(?<version>\d+(?:\.\d+)*)'
    )
    if (-not $match.Success) {
        return $null
    }
    return $match.Groups["version"].Value
}

function Resolve-JavaHome {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$ExpectedMajor
    )
    $explicit = [Environment]::GetEnvironmentVariable(
        "BLINDASSIST_JAVA_HOME",
        "Process"
    )
    if (-not [string]::IsNullOrWhiteSpace($explicit)) {
        if (-not (Test-Path -LiteralPath $explicit -PathType Container)) {
            Stop-Environment "BLINDASSIST_JAVA_HOME does not exist: $explicit"
        }
        $version = Get-JavaVersion -JavaHome $explicit
        if ($null -eq $version -or $version.Split('.')[0] -ne $ExpectedMajor) {
            Stop-Environment (
                "BLINDASSIST_JAVA_HOME must contain JDK $ExpectedMajor; " +
                "actual='$version', path='$explicit'."
            )
        }
        return [pscustomobject]@{
            home = (Resolve-Path -LiteralPath $explicit).Path
            version = $version
        }
    }

    $candidates = [System.Collections.Generic.List[string]]::new()
    $repoJdkRoot = Join-Path $RepoRoot ".jdk"
    $candidates.Add((Join-Path $repoJdkRoot "jdk17.0.19_10"))
    if (Test-Path -LiteralPath $repoJdkRoot -PathType Container) {
        Get-ChildItem -LiteralPath $repoJdkRoot -Directory |
            Sort-Object Name |
            ForEach-Object { $candidates.Add($_.FullName) }
    }
    $currentJavaHome = [Environment]::GetEnvironmentVariable(
        "JAVA_HOME",
        "Process"
    )
    if (-not [string]::IsNullOrWhiteSpace($currentJavaHome)) {
        $candidates.Add($currentJavaHome)
    }
    $candidates.Add("E:\codex-tools\tools\jdk17.0.19_10")
    $candidates.Add("C:\Program Files\Android\Android Studio\jbr")

    $checked = [System.Collections.Generic.List[string]]::new()
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (
            [string]::IsNullOrWhiteSpace($candidate) -or
            -not (Test-Path -LiteralPath $candidate -PathType Container)
        ) {
            continue
        }
        $resolved = (Resolve-Path -LiteralPath $candidate).Path
        $version = Get-JavaVersion -JavaHome $resolved
        $checked.Add("$resolved=$version")
        if ($null -ne $version -and $version.Split('.')[0] -eq $ExpectedMajor) {
            return [pscustomobject]@{
                home = $resolved
                version = $version
            }
        }
    }
    Stop-Environment (
        "No usable JDK $ExpectedMajor was found. Checked: " +
        ($checked -join "; ") +
        ". Install the project toolchain or set BLINDASSIST_JAVA_HOME."
    )
}

function Convert-LocalPropertiesPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    return ($Value.Trim() -replace '\\:', ':' -replace '\\\\', '\')
}

function Resolve-AndroidSdk {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$CompileSdk
    )
    $localProperties = Join-Path $RepoRoot "local.properties"
    $localSdk = $null
    if (Test-Path -LiteralPath $localProperties -PathType Leaf) {
        $content = Get-Content -LiteralPath $localProperties -Raw
        $match = [regex]::Match(
            $content,
            '(?m)^\s*sdk\.dir\s*=\s*(?<path>.+?)\s*$'
        )
        if ($match.Success) {
            $localSdk = Convert-LocalPropertiesPath $match.Groups["path"].Value
            if (-not [IO.Path]::IsPathRooted($localSdk)) {
                $localSdk = Join-Path $RepoRoot $localSdk
            }
        }
    }

    $explicit = [Environment]::GetEnvironmentVariable(
        "BLINDASSIST_ANDROID_SDK_ROOT",
        "Process"
    )
    if (-not [string]::IsNullOrWhiteSpace($explicit)) {
        if ($null -ne $localSdk) {
            $explicitFull = [IO.Path]::GetFullPath($explicit).TrimEnd('\', '/')
            $localFull = [IO.Path]::GetFullPath($localSdk).TrimEnd('\', '/')
            if (-not $explicitFull.Equals(
                $localFull,
                [StringComparison]::OrdinalIgnoreCase
            )) {
                Stop-Environment (
                    "BLINDASSIST_ANDROID_SDK_ROOT conflicts with " +
                    "local.properties sdk.dir. Gradle would use '$localFull', " +
                    "not '$explicitFull'."
                )
            }
        }
        $candidate = $explicit
    } elseif ($null -ne $localSdk) {
        $candidate = $localSdk
    } else {
        $candidate = @(
            (Join-Path $RepoRoot ".android-sdk"),
            [Environment]::GetEnvironmentVariable("ANDROID_SDK_ROOT", "Process"),
            [Environment]::GetEnvironmentVariable("ANDROID_HOME", "Process"),
            "E:\codex-tools\projects\blindassist\toolchain\android-sdk"
        ) |
            Where-Object {
                -not [string]::IsNullOrWhiteSpace($_) -and
                (Test-Path -LiteralPath $_ -PathType Container)
            } |
            Select-Object -First 1
    }

    if (
        [string]::IsNullOrWhiteSpace($candidate) -or
        -not (Test-Path -LiteralPath $candidate -PathType Container)
    ) {
        Stop-Environment (
            "Android SDK is missing. Restore local.properties/.android-sdk " +
            "or set BLINDASSIST_ANDROID_SDK_ROOT."
        )
    }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    $androidJar = Join-Path $resolved "platforms\android-$CompileSdk\android.jar"
    if (-not (Test-Path -LiteralPath $androidJar -PathType Leaf)) {
        Stop-Environment (
            "Android SDK Platform $CompileSdk is missing under '$resolved'."
        )
    }
    return $resolved
}

function Resolve-StateDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$ExplicitVariable,
        [Parameter(Mandatory = $true)][string[]]$Candidates
    )
    $explicit = [Environment]::GetEnvironmentVariable(
        $ExplicitVariable,
        "Process"
    )
    if (-not [string]::IsNullOrWhiteSpace($explicit)) {
        $selected = $explicit
    } else {
        $selected = $Candidates |
            Where-Object {
                -not [string]::IsNullOrWhiteSpace($_) -and
                (Test-Path -LiteralPath $_ -PathType Container)
            } |
            Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($selected)) {
            $selected = $Candidates[-1]
        }
    }
    if (-not (Test-Path -LiteralPath $selected -PathType Container)) {
        New-Item -ItemType Directory -Path $selected -Force | Out-Null
    }
    return (Resolve-Path -LiteralPath $selected).Path
}

function Get-AdbDevices {
    param([Parameter(Mandatory = $true)][string]$Adb)
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Adb
    $startInfo.Arguments = "devices -l"
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        Stop-Environment "Could not start adb: $Adb"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit(10000)) {
        try {
            $process.Kill($true)
        } catch {
            $process.Kill()
        }
        Stop-Environment "adb devices -l timed out after 10 seconds."
    }
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) {
        Stop-Environment (
            "adb devices -l failed with exit code $($process.ExitCode): " +
            $stderr.Trim()
        )
    }
    return @($stdout -split [Environment]::NewLine)
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$catalogPath = Join-Path $repoRoot "gradle\libs.versions.toml"
$wrapperPropertiesPath = Join-Path (
    Join-Path $repoRoot "gradle\wrapper"
) "gradle-wrapper.properties"
$wrapper = Join-Path $repoRoot "gradlew.bat"
foreach ($required in @($catalogPath, $wrapperPropertiesPath, $wrapper)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        Stop-Environment "Required build file is missing: $required"
    }
}

$catalog = Get-Content -LiteralPath $catalogPath -Raw
$expectedJavaMajor = Get-DeclaredVersion $catalog "jvmTarget"
$compileSdk = Get-DeclaredVersion $catalog "compileSdk"
$agpVersion = Get-DeclaredVersion $catalog "agp"
$kotlinVersion = Get-DeclaredVersion $catalog "kotlin"
$wrapperProperties = Get-Content -LiteralPath $wrapperPropertiesPath -Raw
$wrapperMatch = [regex]::Match(
    $wrapperProperties,
    'gradle-(?<version>\d+(?:\.\d+)+)-bin\.zip'
)
if (-not $wrapperMatch.Success) {
    Stop-Environment "Could not determine Gradle version from wrapper properties."
}
$expectedGradleVersion = $wrapperMatch.Groups["version"].Value

$java = Resolve-JavaHome $repoRoot $expectedJavaMajor
$androidSdk = Resolve-AndroidSdk $repoRoot $compileSdk
$gradleUserHome = Resolve-StateDirectory (
    "BLINDASSIST_GRADLE_USER_HOME"
) @(
    "E:\codex-tools\projects\blindassist\state\gradle",
    (Join-Path $repoRoot "artifacts.local\gradle-home")
)
$androidUserHome = Resolve-StateDirectory (
    "BLINDASSIST_ANDROID_USER_HOME"
) @(
    (Join-Path $repoRoot ".android-home")
)

$env:JAVA_HOME = $java.home
$env:ANDROID_HOME = $androidSdk
$env:ANDROID_SDK_ROOT = $androidSdk
$env:ANDROID_USER_HOME = $androidUserHome
$env:GRADLE_USER_HOME = $gradleUserHome
$kotlinHome = Join-Path $repoRoot ".kotlin-home"
if (Test-Path -LiteralPath $kotlinHome -PathType Container) {
    $env:KOTLIN_HOME = (Resolve-Path -LiteralPath $kotlinHome).Path
}
$pathPrefixes = @(
    (Join-Path $env:JAVA_HOME "bin"),
    (Join-Path $androidSdk "platform-tools")
)
$existingPath = @($env:PATH -split ';') |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
$env:PATH = (@($pathPrefixes) + $existingPath | Select-Object -Unique) -join ';'

$effectiveGradleArguments = @($GradleArguments)
$connectedTaskRequested = $effectiveGradleArguments |
    Where-Object { $_ -match '(^|:)connected[^:]*AndroidTest$' } |
    Select-Object -First 1
if (
    -not [string]::IsNullOrWhiteSpace($AndroidSerial) -or
    $null -ne $connectedTaskRequested
) {
    $RequireDevice = $true
}

$selectedSerial = $null
if ($RequireDevice) {
    $adb = Join-Path $androidSdk "platform-tools\adb.exe"
    if (-not (Test-Path -LiteralPath $adb -PathType Leaf)) {
        Stop-Environment "Android platform-tools adb.exe is missing: $adb"
    }
    $deviceRows = Get-AdbDevices $adb |
        Where-Object { $_ -match '^\S+\s+device(?:\s|$)' }
    if (-not [string]::IsNullOrWhiteSpace($AndroidSerial)) {
        $selectedRow = $deviceRows |
            Where-Object { ($_ -split '\s+')[0] -eq $AndroidSerial } |
            Select-Object -First 1
        if ($null -eq $selectedRow) {
            Stop-Environment (
                "Requested Android device '$AndroidSerial' is not connected " +
                "and authorized."
            )
        }
        $selectedSerial = $AndroidSerial
    } else {
        if (@($deviceRows).Count -eq 0) {
            Stop-Environment (
                "A connected/authorized Android device is required, but none " +
                "was found."
            )
        }
        if (@($deviceRows).Count -gt 1) {
            $serials = @($deviceRows | ForEach-Object { ($_ -split '\s+')[0] })
            Stop-Environment (
                "Multiple Android devices are connected ($($serials -join ', ')). " +
                "Pass -AndroidSerial explicitly."
            )
        }
        $selectedSerial = (@($deviceRows)[0] -split '\s+')[0]
    }
    $env:ANDROID_SERIAL = $selectedSerial
}

Push-Location $repoRoot
try {
    $probeOutput = @(
        & $wrapper --version --no-daemon --console=plain 2>&1
    )
    $probeExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($probeExitCode -ne 0) {
    Stop-Environment (
        "Gradle wrapper probe failed with exit code $probeExitCode. " +
        (($probeOutput | Select-Object -Last 8) -join " | ")
    )
}
$actualGradleMatch = [regex]::Match(
    ($probeOutput -join [Environment]::NewLine),
    '(?m)^Gradle\s+(?<version>\S+)\s*$'
)
if (
    -not $actualGradleMatch.Success -or
    $actualGradleMatch.Groups["version"].Value -ne $expectedGradleVersion
) {
    Stop-Environment (
        "Gradle wrapper version mismatch: expected $expectedGradleVersion, " +
        "actual '$($actualGradleMatch.Groups['version'].Value)'."
    )
}

$summary = [ordered]@{
    status = "ENVIRONMENT_READY"
    repo_root = $repoRoot
    java_home = $env:JAVA_HOME
    java_version = $java.version
    android_sdk_root = $androidSdk
    compile_sdk = $compileSdk
    gradle_user_home = $env:GRADLE_USER_HOME
    gradle_version = $expectedGradleVersion
    agp_version = $agpVersion
    kotlin_version = $kotlinVersion
    device_required = [bool]$RequireDevice
    android_serial = $selectedSerial
}
Write-Output ($summary | ConvertTo-Json -Compress)

if ($PreflightOnly) {
    return
}
if ($effectiveGradleArguments.Count -eq 0) {
    Stop-Usage (
        "Pass one or more Gradle tasks/arguments, or use -PreflightOnly."
    )
}
if (-not ($effectiveGradleArguments -match '^--(?:no-)?daemon$')) {
    $effectiveGradleArguments += "--no-daemon"
}
if (-not ($effectiveGradleArguments -match '^--console=')) {
    $effectiveGradleArguments += "--console=plain"
}
if (-not ($effectiveGradleArguments -match '^--max-workers(?:=|$)')) {
    $effectiveGradleArguments += "--max-workers=2"
}

Push-Location $repoRoot
try {
    & $wrapper @effectiveGradleArguments
    $gradleExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($null -eq $gradleExitCode) {
    $gradleExitCode = 0
}
if ($gradleExitCode -ne 0) {
    [Console]::Error.WriteLine(
        "GRADLE_COMMAND_FAILED: exit_code=$gradleExitCode; " +
        "environment_preflight=PASS"
    )
    exit $gradleExitCode
}
Write-Output "GRADLE_COMMAND_COMPLETE: exit_code=0"
