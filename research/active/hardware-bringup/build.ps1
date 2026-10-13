param(
    [ValidateSet('i2c_probe','tof_reader','tof_cnh','tof_cnh_h3','tof_cnh_h3_wifi','tof_cnh_diag','atom_camera','tof_wifi','atom_wifi')][string]$Sketch = 'tof_reader',
    [ValidateSet(0,32768)][int]$AtomTcpSendBufferBytes = 0,
    [switch]$AtomCameraEarlyEoi
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$configPath = Join-Path $repo 'artifacts.local/hardware-bringup/local-config.json'
if (-not (Test-Path -LiteralPath $configPath)) { throw 'Run prepare.ps1 first.' }
$cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'vendor-lock.json') -Raw | ConvertFrom-Json
$src = Join-Path $cfg.artifact_root "sketches/$Sketch"
$build = Join-Path $cfg.artifact_root "build/$Sketch"
New-Item -ItemType Directory -Path $src,$build -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "firmware/$Sketch/$Sketch.ino") -Destination $src
if ($Sketch -in @('tof_wifi','tof_cnh_h3_wifi','atom_wifi')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'firmware/demo_wifi.h') -Destination $src
}
if ($Sketch -eq 'atom_wifi') {
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'firmware/atom_wifi') -Filter '*.h' -File |
        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $src }
}
$hashes = [ordered]@{}
if ($Sketch -in @('tof_reader','tof_cnh','tof_cnh_h3','tof_cnh_h3_wifi','tof_cnh_diag','tof_wifi')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'firmware/platform.h'),(Join-Path $PSScriptRoot 'firmware/platform.cpp') -Destination $src
    foreach ($file in $lock.files) {
        # Baseline needs no CNH implementation; headers remain identical to the vendor package.
        if ($Sketch -in @('tof_reader','tof_wifi') -and $file.path -eq 'src/vl53lmz_plugin_cnh.c') { continue }
        $path = Join-Path (Join-Path $cfg.vendor_package $lock.source_subdirectory) $file.path
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $file.sha256) { throw "Driver hash mismatch: $($file.path)" }
        Copy-Item -LiteralPath $path -Destination $src
        $hashes[$file.path] = $hash
    }
}
$cliConfig = Join-Path $cfg.artifact_root 'arduino-cli.json'
@{ directories = @{ data=$cfg.arduino_data; downloads=(Join-Path $cfg.artifact_root 'downloads'); user=(Join-Path $cfg.artifact_root 'arduino-user') } } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $cliConfig -Encoding utf8
$fqbn = $cfg.fqbn
if ($Sketch -in @('atom_camera','atom_wifi')) {
    $fqbn = 'm5stack:esp32:m5stack_atoms3r:PSRAM=opi,FlashMode=qio,FlashSize=8M,PartitionScheme=default_8MB,USBMode=default,CDCOnBoot=cdc'
}
$extraBuildArgs = @()
$lwipBuild = $null
$cameraBuild = $null
$overrideLibraries = @()
if ($AtomTcpSendBufferBytes -ne 0) {
    if ($Sketch -ne 'atom_wifi') { throw 'AtomTcpSendBufferBytes applies only to atom_wifi.' }
    $lwipOutput = Join-Path $cfg.artifact_root "build/atom_lwip_snd$AtomTcpSendBufferBytes"
    & $cfg.python (Join-Path $PSScriptRoot 'host/rebuild_atom_lwip.py') --output-dir $lwipOutput --send-buffer $AtomTcpSendBufferBytes
    if ($LASTEXITCODE) { throw 'Isolated lwIP rebuild failed.' }
    $lwipLibrary = [IO.Path]::GetFullPath((Join-Path $lwipOutput 'liblwip.a')).Replace('\','/')
    $overrideLibraries += $lwipLibrary
    $lwipBuild = @{ send_buffer_bytes=$AtomTcpSendBufferBytes; library=$lwipLibrary;
        library_sha256=(Get-FileHash -LiteralPath $lwipLibrary).Hash.ToLowerInvariant() }
}
if ($AtomCameraEarlyEoi) {
    if ($Sketch -ne 'atom_wifi') { throw 'AtomCameraEarlyEoi applies only to atom_wifi.' }
    $cameraOutput = Join-Path $cfg.artifact_root 'build/atom_camera_early_eoi'
    & $cfg.python (Join-Path $PSScriptRoot 'host/rebuild_atom_camera.py') --output-dir $cameraOutput
    if ($LASTEXITCODE) { throw 'Isolated camera rebuild failed.' }
    $cameraLibrary = [IO.Path]::GetFullPath((Join-Path $cameraOutput 'libespressif__esp32-camera.a')).Replace('\','/')
    $overrideLibraries += $cameraLibrary
    $cameraBuild = @{ native_first_eoi=$true; library=$cameraLibrary;
        library_sha256=(Get-FileHash -LiteralPath $cameraLibrary).Hash.ToLowerInvariant() }
}
if ($overrideLibraries.Count) {
    $quotedLibraries = ($overrideLibraries | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $extraBuildArgs = @('--build-property', "build.extra_libs=-Wl,--whole-archive $quotedLibraries -Wl,--no-whole-archive")
}
& $cfg.arduino_cli compile --config-file $cliConfig --fqbn $fqbn --build-path $build @extraBuildArgs $src 2>&1 |
    Tee-Object -FilePath (Join-Path $build 'compile.log')
if ($LASTEXITCODE) { throw "Compilation failed: $Sketch" }
if ($lwipBuild) {
    $map = Get-Content -LiteralPath (Join-Path $build "$Sketch.ino.map") -Raw
    $linkedLibraries = @([regex]::Matches($map, '([^\s()]*liblwip\.a)\(') |
        ForEach-Object { $_.Groups[1].Value.Replace('\','/') } | Sort-Object -Unique)
    if ($linkedLibraries.Count -ne 1 -or $linkedLibraries[0] -ne $lwipLibrary) {
        throw "Unexpected or mixed lwIP link: $linkedLibraries"
    }
}
if ($cameraBuild) {
    $map = Get-Content -LiteralPath (Join-Path $build "$Sketch.ino.map") -Raw
    $linkedLibraries = @([regex]::Matches($map, '([^\s()]*libespressif__esp32-camera\.a)\(') |
        ForEach-Object { $_.Groups[1].Value.Replace('\','/') } | Sort-Object -Unique)
    if ($linkedLibraries.Count -ne 1 -or $linkedLibraries[0] -ne $cameraLibrary) {
        throw "Unexpected or mixed camera link: $linkedLibraries"
    }
}
$binary = Join-Path $build "$Sketch.ino.bin"
$sourceHashes = @{}
Get-ChildItem -LiteralPath $src -File | ForEach-Object { $sourceHashes[$_.Name]=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
@{ sketch=$Sketch; fqbn=$fqbn; source_sha256=$sourceHashes; vendor_sha256=$hashes; lwip_override=$lwipBuild; camera_override=$cameraBuild;
   app_sha256=(Get-FileHash -LiteralPath $binary).Hash.ToLowerInvariant(); app_bytes=(Get-Item -LiteralPath $binary).Length;
   compiled_utc=[DateTime]::UtcNow.ToString('o'); device_operation='none'; binary=$binary } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Encoding utf8
Write-Output "Built $binary. No upload performed."
