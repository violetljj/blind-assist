"""Transfer hashed task sources and dispatch one bounded worker capture.

The dispatch receipt is an acknowledgement, never a capture PASS.
"""
from __future__ import annotations
import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from cnh_route_control import HERE, REPO, ROOT
from cnh_route_spec import sha, write


def remote(script, name):
    path=ROOT/'worker-scripts'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(script,encoding='utf-8')
    run=subprocess.run(['pwsh','-NoProfile','-File',str(REPO/'tools/remote_worker.ps1'),
        '-Action','Exec','-ScriptFile',str(path)],cwd=REPO,text=True,capture_output=True)
    (path.with_suffix('.stdout.log')).write_text(run.stdout,encoding='utf-8')
    (path.with_suffix('.stderr.log')).write_text(run.stderr,encoding='utf-8')
    if run.returncode:
        raise RuntimeError(f'Remote exit {run.returncode}; inspect {path}. Do not redispatch uncertain jobs.')
    print(run.stdout)
    return run.stdout


def dispatch(spec, attempt, timeout, pilot=False):
    source=json.loads(Path(spec).read_text(encoding='utf-8-sig'))
    if pilot:
        from cnh_route_pilot import require_benchmark_source
        require_benchmark_source(source)
    else:
        from cnh_route_capture import validate_spec
        validate_spec(source)
    if not attempt.replace('-','').isalnum():
        raise ValueError('Invalid attempt name')
    journal=__import__('os').environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal:
        raise RuntimeError('Dispatch through tools/ba.ps1 run research-ue -RunSpec')
    receipt=ROOT/'dispatch'/f'{attempt}.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    files=[HERE/n for n in ['cnh_route_capture.py','cnh_route_launch.py','ue_capture_readiness.py','ue_pair_export.py']]
    if pilot:
        files += [HERE/n for n in ['cnh_route_pilot.py','cnh_route_geometry.py','cnh_route_pack.py']]
    files += [HERE.parent/'unreal/street_process_lifecycle.py',spec]
    manifest={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    lines=["$ErrorActionPreference='Stop'",
        f"$taskRoot=Join-Path $env:BLINDASSIST_ARTIFACTS 'work/cnh-route-comparison-20260924/{attempt}'",
        "if(Test-Path -LiteralPath $taskRoot){throw 'Fresh attempt required'}",
        "New-Item -ItemType Directory -Path (Join-Path $taskRoot 'tasksrc') | Out-Null"]
    for p in files:
        payload=base64.b64encode(p.read_bytes()).decode()
        lines += [f"$target=Join-Path $taskRoot 'tasksrc/{p.name}'",
            f"[IO.File]::WriteAllBytes($target,[Convert]::FromBase64String('{payload}'))",
            f"if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne '{sha(p)}'){{throw 'Transfer hash mismatch'}}"]
    payload=base64.b64encode(json.dumps(manifest,indent=2).encode()).decode()
    entry='cnh_route_pilot.py' if pilot else 'cnh_route_launch.py'
    lines += [f"[IO.File]::WriteAllBytes((Join-Path $taskRoot 'source-hashes.json'),[Convert]::FromBase64String('{payload}'))",
        f"python \"$env:BLINDASSIST_WORKER_ROOT/scripts/run_job.py\" --job 'cnh-route-{attempt}' -- python (Join-Path $taskRoot 'tasksrc/{entry}') --spec (Join-Path $taskRoot 'tasksrc/{spec.name}') --output (Join-Path $taskRoot 'capture') --timeout-sec {timeout}",
        "if($LASTEXITCODE -ne 0){throw 'Dispatch failed; inspect job receipt before any retry'}"]
    result=dict(status='DISPATCHING',attempt=attempt,source_hashes=manifest,
        worker_relative=f'work/cnh-route-comparison-20260924/{attempt}',timeout_sec=timeout,
        scope='ENGINEERING_NATIVE_CAPTURE_NOT_MODEL_EVALUATION')
    write(receipt,result)
    try:
        result['acknowledgement']=remote('\n'.join(lines),attempt+'-dispatch.ps1')
        result['status']='ACKNOWLEDGED_NOT_COMPLETED'
    except Exception as exc:
        result.update(status='DISPATCH_UNCERTAIN_INSPECT_BEFORE_RETRY',error=str(exc))
        raise
    finally:
        write(receipt,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--attempt',required=True)
    p.add_argument('--timeout-sec',type=int,default=600)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    print(json.dumps(dispatch(a.spec,a.attempt,a.timeout_sec,a.pilot),indent=2))
