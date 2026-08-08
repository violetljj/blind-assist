[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$QairtRoot = $env:QAIRT_SDK_ROOT,
    [string]$HexagonSdkRoot = $env:HEXAGON_SDK_ROOT,
    [string]$AndroidNdkRoot = $env:ANDROID_NDK_ROOT,
    [string]$Python310 = 'E:\codex-tools\tools\venvs\qairt310\Scripts\python.exe',
    [ValidateSet('v73', 'v75')]
    [string]$TargetArch = 'v73'
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$hftfRoot = (Resolve-Path (Join-Path $scriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot '..\..\..\..\..')).Path
$artifactsRoot = (Resolve-Path (Join-Path $repoRoot 'artifacts.local')).Path
$outputParent = Split-Path -Parent $OutputRoot
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
$resolvedOutputParent = (Resolve-Path $outputParent).Path
if (-not $resolvedOutputParent.StartsWith($artifactsRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must be under $artifactsRoot"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "OutputRoot already exists; use a fresh evidence directory: $OutputRoot"
}

$required = @(
    $QairtRoot,
    $HexagonSdkRoot,
    $AndroidNdkRoot,
    $Python310,
    (Join-Path $QairtRoot 'bin\x86_64-windows-msvc\qnn-op-package-generator'),
    (Join-Path $HexagonSdkRoot 'tools\HEXAGON_Tools\8.7.06\Tools\bin\hexagon-clang++.exe')
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required tool or path: $path" }
}

$xml = Join-Path $hftfRoot 'depthart_selective_scan_op_package.xml'
$kernel = Join-Path $scriptRoot 'depthart_selective_scan_htp_reference.cpp'
$generator = Join-Path $QairtRoot 'bin\x86_64-windows-msvc\qnn-op-package-generator'
$env:HEXAGON_SDK_ROOT = $HexagonSdkRoot
$env:PYTHONPATH = "$QairtRoot\lib\python;$QairtRoot\lib\python\qti\aisw\converters\common\windows-x86_64"
$env:PATH = "$QairtRoot\bin\x86_64-windows-msvc;$QairtRoot\lib\x86_64-windows-msvc;$env:PATH"
& $Python310 $generator -p $xml -o $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "qnn-op-package-generator failed: $LASTEXITCODE" }

$packageRoot = Join-Path $OutputRoot 'DepthArtSelectiveScanPackage'
Copy-Item -LiteralPath $kernel -Destination (Join-Path $packageRoot 'src\ops\SelectiveScan.cpp') -Force
$hexagonCxx = Join-Path $HexagonSdkRoot 'tools\HEXAGON_Tools\8.7.06\Tools\bin\hexagon-clang++.exe'
$qnnInclude = ($QairtRoot -replace '\\', '/') + '/include/QNN'
$hexagonUnix = $HexagonSdkRoot -replace '\\', '/'
$computeTarget = "compute$TargetArch"
$hexBuild = Join-Path $packageRoot "build\hexagon-$TargetArch-manual"
New-Item -ItemType Directory -Path $hexBuild -Force | Out-Null
$hexCommon = @(
    '-std=c++17', "-I$qnnInclude", '-fPIC', '-Wall', '-Wreorder',
    '-Wno-missing-braces', '-Wno-unused-function', '-Werror', '-Wno-format',
    '-Wno-unused-command-line-argument', '-fvisibility=default', '-stdlib=libc++',
    '-mhvx', '-mhvx-length=128B', '-mhmx', '-DUSE_OS_QURT', '-O2',
    '-Wno-reorder', '-DPREPARE_DISABLED', "-m$TargetArch",
    "-I$hexagonUnix/rtos/qurt/$computeTarget/include/qurt",
    "-I$hexagonUnix/rtos/qurt/$computeTarget/include/posix",
    "-I$hexagonUnix/incs", "-I$hexagonUnix/incs/stddef",
    '-DTHIS_PKG_NAME=DepthArtSelectiveScanPackage', '-MMD', '-c'
)
$interface = Join-Path $packageRoot 'src\DepthArtSelectiveScanPackageInterface.cpp'
$op = Join-Path $packageRoot 'src\ops\SelectiveScan.cpp'
$hexInterfaceObj = Join-Path $hexBuild 'DepthArtSelectiveScanPackageInterface.o'
$hexOpObj = Join-Path $hexBuild 'SelectiveScan.o'
& $hexagonCxx @hexCommon $interface -o $hexInterfaceObj
if ($LASTEXITCODE -ne 0) { throw "v73 interface compile failed: $LASTEXITCODE" }
& $hexagonCxx @hexCommon $op -o $hexOpObj
if ($LASTEXITCODE -ne 0) { throw "v73 kernel compile failed: $LASTEXITCODE" }
$hexLibrary = Join-Path $hexBuild 'libQnnDepthArtSelectiveScanPackage.so'
& $hexagonCxx -fPIC -std=c++17 -g -shared -o $hexLibrary $hexInterfaceObj $hexOpObj
if ($LASTEXITCODE -ne 0) { throw "v73 package link failed: $LASTEXITCODE" }

$ndkPrebuilt = Join-Path $AndroidNdkRoot 'toolchains\llvm\prebuilt\windows-x86_64'
$androidCxx = Join-Path $ndkPrebuilt 'bin\clang++.exe'
$sysroot = Join-Path $ndkPrebuilt 'sysroot'
$androidBuild = Join-Path $packageRoot 'build\aarch64-android-manual'
New-Item -ItemType Directory -Path $androidBuild -Force | Out-Null
$androidCommon = @(
    '--target=aarch64-none-linux-android21', "--sysroot=$sysroot", '-stdlib=libc++',
    '-static-libstdc++', '-std=c++17', "-I$qnnInclude", '-fPIC', '-Wall',
    '-Wreorder', '-Wno-missing-braces', '-Wno-unused-function', '-Werror',
    '-Wno-format', '-Wno-unused-command-line-argument', '-fvisibility=default',
    '-D__HVXDBL__', "-I$hexagonUnix/tools/HEXAGON_Tools/8.7.06/Tools/libnative/include",
    '-DUSE_OS_LINUX', '-DANDROID', '-fomit-frame-pointer', '-Wno-invalid-offsetof',
    '-Wno-unused-variable', '-Wno-unused-parameter', '-Wno-sign-compare',
    '-Wno-unused-private-field', '-Wno-ignored-qualifiers',
    '-Wno-missing-field-initializers', '-DTHIS_PKG_NAME=DepthArtSelectiveScanPackage',
    '-MMD', '-c'
)
$androidInterfaceObj = Join-Path $androidBuild 'DepthArtSelectiveScanPackageInterface.o'
$androidOpObj = Join-Path $androidBuild 'SelectiveScan.o'
& $androidCxx @androidCommon $interface -o $androidInterfaceObj
if ($LASTEXITCODE -ne 0) { throw "aarch64 interface compile failed: $LASTEXITCODE" }
& $androidCxx @androidCommon $op -o $androidOpObj
if ($LASTEXITCODE -ne 0) { throw "aarch64 kernel compile failed: $LASTEXITCODE" }
$androidLibrary = Join-Path $androidBuild 'libQnnDepthArtSelectiveScanPackage.so'
$qnnAndroidLib = ($QairtRoot -replace '\\', '/') + '/lib/aarch64-android'
& $androidCxx '--target=aarch64-none-linux-android21' "--sysroot=$sysroot" `
    -stdlib=libc++ -static-libstdc++ -fPIC -std=c++17 -g -shared -o $androidLibrary `
    $androidInterfaceObj $androidOpObj "-L$qnnAndroidLib" -lQnnHtp -lQnnHtpPrepare
if ($LASTEXITCODE -ne 0) { throw "aarch64 package link failed: $LASTEXITCODE" }

$receipt = [ordered]@{
    schema_version = 1
    status = 'COMPILED_NOT_RUNTIME_EVALUATED'
    qairt_root = $QairtRoot
    hexagon_sdk_root = $HexagonSdkRoot
    android_ndk_root = $AndroidNdkRoot
    package_interface = 'DepthArtSelectiveScanPackageInterfaceProvider'
    binaries = @(
    [ordered]@{ target = "hexagon-$TargetArch"; path = $hexLibrary; bytes = (Get-Item $hexLibrary).Length; sha256 = (Get-FileHash $hexLibrary -Algorithm SHA256).Hash },
        [ordered]@{ target = 'aarch64-android'; path = $androidLibrary; bytes = (Get-Item $androidLibrary).Length; sha256 = (Get-FileHash $androidLibrary -Algorithm SHA256).Hash }
    )
    authority = 'Compilation only; no QNN context, HTP execution, parity, latency, thermal, Android, safety, or production claim.'
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputRoot 'build-receipt.json') -Encoding utf8
$receipt
