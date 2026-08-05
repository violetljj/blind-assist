param(
    [Parameter(Mandatory = $true)]
    [string]$Port,
    [int]$DurationSeconds = 60,
    [int]$BaudRate = 115200,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$artifactRoot = Join-Path $repoRoot 'artifacts.local'
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutputPath = Join-Path $artifactRoot "evidence\tof4m\tof4m-$stamp.jsonl"
}
$absoluteOutput = [System.IO.Path]::GetFullPath($OutputPath)
$absoluteArtifactRoot = [System.IO.Path]::GetFullPath($artifactRoot).TrimEnd('\') + '\'
if (-not $absoluteOutput.StartsWith($absoluteArtifactRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputPath must remain under $artifactRoot"
}

$parent = Split-Path -Parent $absoluteOutput
New-Item -ItemType Directory -Force -Path $parent | Out-Null
if (Test-Path -LiteralPath $absoluteOutput) {
    throw "Refusing to overwrite existing capture: $absoluteOutput"
}

$serial = [System.IO.Ports.SerialPort]::new($Port, $BaudRate)
$serial.NewLine = "`n"
$serial.ReadTimeout = 1000
$writer = [System.IO.StreamWriter]::new($absoluteOutput, $false, [System.Text.UTF8Encoding]::new($false))
$startedUtc = [DateTimeOffset]::UtcNow
$deadline = [DateTime]::UtcNow.AddSeconds($DurationSeconds)
$accepted = 0
$rejected = 0

try {
    $serial.Open()
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $line = $serial.ReadLine().Trim()
        } catch [System.TimeoutException] {
            continue
        }
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $row = $line | ConvertFrom-Json -ErrorAction Stop
            if ($null -eq $row.schema) {
                throw 'missing schema'
            }
            $writer.WriteLine($line)
            $writer.Flush()
            $accepted++
        } catch {
            $rejected++
            Write-Warning "Discarded non-JSONL serial line: $line"
        }
    }
} finally {
    $writer.Dispose()
    if ($serial.IsOpen) {
        $serial.Close()
    }
    $serial.Dispose()
}

$endedUtc = [DateTimeOffset]::UtcNow
$sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $absoluteOutput).Hash.ToLowerInvariant()
$receiptPath = "$absoluteOutput.receipt.json"
$receipt = [ordered]@{
    schema = 'blindassist_atoms3r_tof4m_capture_receipt_r0'
    port = $Port
    baud_rate = $BaudRate
    requested_duration_seconds = $DurationSeconds
    started_utc = $startedUtc.ToString('o')
    ended_utc = $endedUtc.ToString('o')
    accepted_json_lines = $accepted
    rejected_serial_lines = $rejected
    capture_path = $absoluteOutput
    capture_sha256 = $sha256
    evidence_role = 'DEVELOPMENT_DEVICE_CAPTURE_UNVALIDATED'
}
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $receiptPath -Encoding utf8
Write-Output $absoluteOutput
Write-Output $receiptPath
