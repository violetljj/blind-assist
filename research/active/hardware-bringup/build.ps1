param([ValidateSet('i2c_probe','tof_reader','tof_cnh','tof_cnh_diag','atom_camera','tof_wifi','atom_wifi')][string]$Sketch = 'tof_reader')
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
if ($Sketch -in @('tof_wifi','atom_wifi')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'firmware/demo_wifi.h') -Destination $src
}
$hashes = [ordered]@{}
if ($Sketch -in @('tof_reader','tof_cnh','tof_cnh_diag','tof_wifi')) {
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
& $cfg.arduino_cli compile --config-file $cliConfig --fqbn $fqbn --build-path $build $src 2>&1 |
    Tee-Object -FilePath (Join-Path $build 'compile.log')
if ($LASTEXITCODE) { throw "Compilation failed: $Sketch" }
$binary = Join-Path $build "$Sketch.ino.bin"
$sourceHashes = @{}
Get-ChildItem -LiteralPath $src -File | ForEach-Object { $sourceHashes[$_.Name]=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
@{ sketch=$Sketch; fqbn=$fqbn; source_sha256=$sourceHashes; vendor_sha256=$hashes;
   app_sha256=(Get-FileHash -LiteralPath $binary).Hash.ToLowerInvariant(); app_bytes=(Get-Item -LiteralPath $binary).Length;
   compiled_utc=[DateTime]::UtcNow.ToString('o'); device_operation='none'; binary=$binary } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Encoding utf8
Write-Output "Built $binary. No upload performed."
