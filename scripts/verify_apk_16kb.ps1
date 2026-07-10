param(
    [Parameter(Mandatory = $true)]
    [Alias("ApkPath", "AabPath")]
    [string]$ArtifactPath,
    [string]$AndroidSdkRoot,
    [string]$BundletoolPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $PSScriptRoot "..\$Path"
}

function Get-LatestBuildTools([string]$SdkRoot) {
    $buildTools = Join-Path $SdkRoot "build-tools"
    if (-not (Test-Path -LiteralPath $buildTools)) {
        throw "Android build-tools not found under $buildTools"
    }
    $latest = Get-ChildItem -LiteralPath $buildTools -Directory |
        Sort-Object { try { [version]$_.Name } catch { [version]"0.0" } } -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No Android build-tools installation found under $buildTools"
    }
    return $latest
}

function Read-UInt16([byte[]]$Bytes, [int]$Offset, [bool]$LittleEndian) {
    $value = [System.BitConverter]::ToUInt16($Bytes, $Offset)
    if ($LittleEndian -eq [System.BitConverter]::IsLittleEndian) { return [uint16]$value }
    return [uint16]((($value -band 0xFF) -shl 8) -bor (($value -shr 8) -band 0xFF))
}

function Read-UInt32([byte[]]$Bytes, [int]$Offset, [bool]$LittleEndian) {
    $slice = $Bytes[$Offset..($Offset + 3)]
    if ($LittleEndian -ne [System.BitConverter]::IsLittleEndian) { [array]::Reverse($slice) }
    return [System.BitConverter]::ToUInt32($slice, 0)
}

function Read-UInt64([byte[]]$Bytes, [int]$Offset, [bool]$LittleEndian) {
    $slice = $Bytes[$Offset..($Offset + 7)]
    if ($LittleEndian -ne [System.BitConverter]::IsLittleEndian) { [array]::Reverse($slice) }
    return [System.BitConverter]::ToUInt64($slice, 0)
}

function Get-ElfLoadAlignments([byte[]]$Bytes, [string]$EntryName) {
    if ($Bytes.Length -lt 52 -or
        $Bytes[0] -ne 0x7F -or $Bytes[1] -ne 0x45 -or
        $Bytes[2] -ne 0x4C -or $Bytes[3] -ne 0x46) {
        throw "$EntryName is not a valid ELF file."
    }

    $elfClass = $Bytes[4]
    $littleEndian = switch ($Bytes[5]) {
        1 { $true }
        2 { $false }
        default { throw "$EntryName has an unsupported ELF byte order." }
    }
    if ($elfClass -eq 1) {
        $programHeaderOffset = [uint64](Read-UInt32 $Bytes 28 $littleEndian)
        $programHeaderEntrySize = [int](Read-UInt16 $Bytes 42 $littleEndian)
        $programHeaderCount = [int](Read-UInt16 $Bytes 44 $littleEndian)
        $alignmentOffset = 28
    } elseif ($elfClass -eq 2) {
        if ($Bytes.Length -lt 64) { throw "$EntryName has a truncated ELF64 header." }
        $programHeaderOffset = Read-UInt64 $Bytes 32 $littleEndian
        $programHeaderEntrySize = [int](Read-UInt16 $Bytes 54 $littleEndian)
        $programHeaderCount = [int](Read-UInt16 $Bytes 56 $littleEndian)
        $alignmentOffset = 48
    } else {
        throw "$EntryName has unsupported ELF class $elfClass."
    }

    if ($programHeaderEntrySize -le 0 -or $programHeaderCount -le 0) {
        throw "$EntryName has no readable ELF program headers."
    }

    $alignments = @()
    for ($index = 0; $index -lt $programHeaderCount; $index++) {
        $headerOffset64 = $programHeaderOffset + ([uint64]$index * [uint64]$programHeaderEntrySize)
        if ($headerOffset64 -gt [int]::MaxValue) { throw "$EntryName program headers exceed supported size." }
        $headerOffset = [int]$headerOffset64
        if ($headerOffset + $programHeaderEntrySize -gt $Bytes.Length) {
            throw "$EntryName has a truncated ELF program header."
        }
        $programType = Read-UInt32 $Bytes $headerOffset $littleEndian
        if ($programType -eq 1) {
            $alignment = if ($elfClass -eq 1) {
                [uint64](Read-UInt32 $Bytes ($headerOffset + $alignmentOffset) $littleEndian)
            } else {
                Read-UInt64 $Bytes ($headerOffset + $alignmentOffset) $littleEndian
            }
            $alignments += $alignment
        }
    }
    if ($alignments.Count -eq 0) { throw "$EntryName has no PT_LOAD segments." }
    return $alignments
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$resolvedArtifact = Resolve-RepoPath $ArtifactPath
if (-not (Test-Path -LiteralPath $resolvedArtifact)) {
    throw "Android artifact not found: $resolvedArtifact"
}
$resolvedArtifact = (Resolve-Path -LiteralPath $resolvedArtifact).Path
$extension = [System.IO.Path]::GetExtension($resolvedArtifact).ToLowerInvariant()
if ($extension -notin @(".apk", ".aab")) {
    throw "Expected an .apk or .aab artifact, got: $resolvedArtifact"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($resolvedArtifact)
try {
    $nativeLibraries = @()
    foreach ($entry in $archive.Entries | Where-Object { $_.FullName -match '(^|/)lib/([^/]+)/([^/]+\.so)$' }) {
        $stream = $entry.Open()
        try {
            $memory = New-Object System.IO.MemoryStream
            try {
                $stream.CopyTo($memory)
                $bytes = $memory.ToArray()
            } finally {
                $memory.Dispose()
            }
        } finally {
            $stream.Dispose()
        }
        $match = [regex]::Match($entry.FullName, '(^|/)lib/([^/]+)/([^/]+\.so)$')
        $alignments = @(Get-ElfLoadAlignments $bytes $entry.FullName)
        $minimumAlignment = ($alignments | Measure-Object -Minimum).Minimum
        $nativeLibraries += [pscustomobject][ordered]@{
            abi = $match.Groups[2].Value
            library = $match.Groups[3].Value
            entry = $entry.FullName
            loadAlignments = @($alignments)
            minimumLoadAlignment = [uint64]$minimumAlignment
            compatible16Kb = ([uint64]$minimumAlignment -ge 16384)
        }
    }
} finally {
    $archive.Dispose()
}

if ($nativeLibraries.Count -eq 0) {
    throw "No native .so libraries found in $resolvedArtifact"
}
$incompatible = @($nativeLibraries | Where-Object { -not $_.compatible16Kb })
if ($incompatible.Count -gt 0) {
    $details = ($incompatible | ForEach-Object { "$($_.abi)/$($_.library)=$($_.minimumLoadAlignment)" }) -join ", "
    throw "Native libraries are not 16KB ELF-aligned: $details"
}

$zipAlignment = $null
$bundleAlignment = $null
if ($extension -eq ".apk") {
    if (-not $AndroidSdkRoot) {
        $AndroidSdkRoot = if ($env:ANDROID_SDK_ROOT) { $env:ANDROID_SDK_ROOT } elseif ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { Join-Path $repoRoot ".android-sdk" }
    }
    $buildTools = Get-LatestBuildTools (Resolve-RepoPath $AndroidSdkRoot)
    $zipalign = Join-Path $buildTools.FullName $(if ($env:OS -eq "Windows_NT") { "zipalign.exe" } else { "zipalign" })
    if (-not (Test-Path -LiteralPath $zipalign)) { throw "zipalign not found in $($buildTools.FullName)" }
    $zipalignOutput = (& $zipalign -c -P 16 -v 4 $resolvedArtifact 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "zipalign 16KB verification failed:`n$zipalignOutput" }
    if ($zipalignOutput -match 'lib[/\\][^\r\n]+\.so \(OK - compressed\)') {
        throw "Compressed native libraries are not allowed; 16KB compatibility must use uncompressed aligned .so files."
    }
    $zipAlignment = "PAGE_ALIGNMENT_16K"
} else {
    if (-not $BundletoolPath) {
        $BundletoolPath = $env:BUNDLETOOL_JAR
    }
    if (-not $BundletoolPath -or -not (Test-Path -LiteralPath (Resolve-RepoPath $BundletoolPath))) {
        throw "AAB verification requires -BundletoolPath or BUNDLETOOL_JAR."
    }
    $resolvedBundletool = (Resolve-Path -LiteralPath (Resolve-RepoPath $BundletoolPath)).Path
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $bundleConfig = (& java -jar $resolvedBundletool dump config "--bundle=$resolvedArtifact" 2>&1) -join "`n"
        $bundletoolExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($bundletoolExitCode -ne 0) { throw "bundletool dump config failed:`n$bundleConfig" }
    if ($bundleConfig -notmatch 'PAGE_ALIGNMENT_16K') {
        throw "AAB bundle config does not declare PAGE_ALIGNMENT_16K."
    }
    $bundleAlignment = "PAGE_ALIGNMENT_16K"
}

[ordered]@{
    artifact = $resolvedArtifact
    artifactType = $extension.TrimStart([char]'.').ToUpperInvariant()
    zipAlignment = $zipAlignment
    bundleAlignment = $bundleAlignment
    nativeLibraryCount = $nativeLibraries.Count
    nativeLibraries = @($nativeLibraries | Sort-Object abi, library)
} | ConvertTo-Json -Depth 6
