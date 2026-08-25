[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('doctor', 'bootstrap', 'test', 'run', 'rebuild')]
    [string]$Command = 'doctor',
    [string]$PythonScript,
    [string[]]$TargetArguments,
    [string[]]$GradleArguments,
    [switch]$RequireDevice,
    [string]$AndroidSerial
)

# Compatibility shim. New work uses tools/ba.ps1 with an explicit profile.
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$entry = Join-Path $repo 'tools/ba.ps1'

switch ($Command) {
    'doctor' { & $entry doctor base }
    'bootstrap' { & $entry setup base }
    'test' { & (Join-Path $PSScriptRoot 'check_project_structure.ps1') }
    'run' {
        if ($GradleArguments) {
            $profile = if ($RequireDevice) { 'device' } else { 'android' }
            & $entry run $profile -- @GradleArguments
        } elseif ($PythonScript) {
            Write-Error 'project.ps1 generic Python run was retired. Use tools/ba.ps1 run <profile> -- <arguments>.'
            exit 2
        } else {
            Write-Error 'run requires -GradleArguments; Python work must select a tools/ba.ps1 profile.'
            exit 2
        }
    }
    'rebuild' {
        Write-Error 'rebuild was retired. Use tools/ba.ps1 setup <profile> for an idempotent repair.'
        exit 2
    }
}

exit $LASTEXITCODE
