[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SourceSeal,
    [Parameter(Mandatory=$true)][string]$ResearchPython,
    [ValidateSet('FIT_ONLY','FINAL_A','FINAL_B')][string[]]$RemainingGroups=@('FIT_ONLY','FINAL_A','FINAL_B'),
    [string]$JoinBridgeAnnex=''
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
$taskRepo=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskSeal=(Resolve-Path -LiteralPath $SourceSeal).Path
$taskRaw=(Resolve-Path -LiteralPath (Join-Path $taskSeal 'raw')).Path
$taskRaw=(& $ResearchPython -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).resolve(strict=True))' $taskRaw).Trim()
if($LASTEXITCODE -ne 0){throw 'Physical artifact root could not be resolved.'}
$taskLease=$null
$taskGroups=@('FIT_ONLY','FINAL_A','FINAL_B')
$taskJoinAnnex=Get-Content -LiteralPath (Join-Path $taskSeal 'join-scope-annex.json') -Raw | ConvertFrom-Json
foreach($taskCode in $taskJoinAnnex.code_sha256.PSObject.Properties){
    $taskCodePath=Join-Path $taskRepo "research/active/dtr-r0/carla/$($taskCode.Name)"
    if((Get-FileHash -LiteralPath $taskCodePath -Algorithm SHA256).Hash -ne $taskCode.Value){throw 'Join implementation differs from pre-RGBD freeze.'}
}
foreach($taskGroup in $taskGroups){
    $taskGroupRoot=Join-Path $taskRaw $taskGroup
    $taskGate=Get-Content -LiteralPath (Join-Path $taskGroupRoot 'roster-source-gate.json') -Raw | ConvertFrom-Json -Depth 100
    if($taskGate.status -ne 'SOURCE_GATE_MET'){throw 'All three source groups must pass before RGB-D capture.'}
    if($taskGroup -notin $RemainingGroups){
        $taskCompleted=Get-Content -LiteralPath (Join-Path $taskGroupRoot 'r1-joined-result.json') -Raw | ConvertFrom-Json
        if($taskCompleted.status -ne 'DTR_R1_FOUR_SENSOR_SOURCE_COMPLETE'){throw 'Omitted groups must already have complete four-sensor joins.'}
        continue
    }
    foreach($taskExisting in @('result.json','r1-joined-result.json','shards/wearable','shards/depth')){
        if(Test-Path -LiteralPath (Join-Path $taskGroupRoot $taskExisting)){throw "Refusing consumed or partial RGB-D stage: $taskGroup/$taskExisting"}
    }
}
try {
    $taskVolume=Get-Item -LiteralPath $taskRaw
    if((Get-PSDrive -Name $taskVolume.PSDrive.Name).Free -lt 80GB){throw 'R1 requires at least80GiB free before capture.'}
    $taskLease=(& (Join-Path $PSScriptRoot 'assert_carla_storage_capacity.ps1') -Action Acquire `
        -ReservationBytes 51539607552 -OutputRoot $taskRaw -LeaseLabel 'DTR-R1-RGBD-completion' | ConvertFrom-Json)
    foreach($taskGroup in $RemainingGroups){
        $taskGroupRoot=Join-Path $taskRaw $taskGroup
        $taskProtocol=Join-Path $taskSeal "$taskGroup/protocol.json"
        try {
            & (Join-Path $PSScriptRoot 'run_dtr_carla_c2_rich_scene.ps1') -RunId $taskGroup `
                -RawEvidenceRoot $taskRaw -Protocol $taskProtocol -Resume `
                -StorageLeaseToken $taskLease.lease_token -CaptureTimeoutSeconds 1800
        } catch {
            # The unchanged generic join's complete-occlusion test is inapplicable
            # to partial S09. R1 finalization below accepts no other failure.
            Write-Output "C2 invocation returned: $($_.Exception.Message)"
        }
        foreach($taskSensor in @('instance','wearable','depth','witness')){
            $taskShardResult=Join-Path $taskGroupRoot "shards/$taskSensor/result.json"
            if(-not (Test-Path -LiteralPath $taskShardResult)){
                throw "Source capture incomplete: $taskGroup/$taskSensor; join and method stages are forbidden."
            }
            $taskShard=Get-Content -LiteralPath $taskShardResult -Raw | ConvertFrom-Json
            if($taskShard.status -ne 'DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE'){
                throw "Source shard did not pass: $taskGroup/$taskSensor"
            }
        }
        if(-not [string]::IsNullOrWhiteSpace($JoinBridgeAnnex)){
            & $ResearchPython (Join-Path $taskRepo 'research/active/dtr-r0/carla/join_dtr_final_roster_source.py') `
                --root $taskGroupRoot --protocol $taskProtocol --annex $JoinBridgeAnnex
        } else {
            if(-not (Test-Path -LiteralPath (Join-Path $taskGroupRoot 'result.json'))){throw "RGB-D capture/join did not complete: $taskGroup"}
            & $ResearchPython (Join-Path $taskRepo 'research/active/dtr-r0/carla/finalize_dtr_final_roster_join.py') `
                --root $taskGroupRoot --protocol $taskProtocol
        }
        if($LASTEXITCODE -ne 0){throw "R1 four-sensor integrity failed: $taskGroup"}
    }
} finally {
    if($null -ne $taskLease){
        & (Join-Path $PSScriptRoot 'assert_carla_storage_capacity.ps1') -Action Release -LeaseToken $taskLease.lease_token
    }
}
