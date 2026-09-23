"""Bounded controller utilities and durable progress for CNH route execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from cnh_route_spec import sha, write

REPO=Path(__file__).resolve().parents[4]
ROOT=REPO/'artifacts.local/work/cnh-route-comparison-20260924'
HERE=Path(__file__).resolve().parent


def progress(stage, status='RUNNING', note='', artifacts=None):
    path=ROOT/'progress.json'
    previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    value=dict(schema='cnh-route-progress-v1',owner='cnh-route-comparison-20260924',
        started_utc=previous.get('started_utc',datetime.now(timezone.utc).isoformat()),
        updated_utc=datetime.now(timezone.utc).isoformat(),stage=stage,status=status,note=note,
        budget=dict(working_days=10,render_gpu_hours=24,fit_gpu_hours=12,evaluation_gpu_hours=12,raw_gib=250),
        plan_sha256=sha(HERE/'CNH_ROUTE_COMPARISON_PLAN_20260924.md'),
        completed_stages=previous.get('completed_stages',[]),
        artifacts=previous.get('artifacts',{})|dict(artifacts or {}),
        final_seven_arm_result=previous.get('final_seven_arm_result','NOT_RUN'),
        pilot20_report=previous.get('pilot20_report','NOT_RUN'))
    if status=='STAGE_COMPLETE' and stage not in value['completed_stages']:
        value['completed_stages'].append(stage)
    write(path,value)
    return value


def make_runspec(stage, command, result, inputs=None, attempt='v1'):
    spec=dict(schema='blindassist-asset-run-v1',id=f'cnh-route-20260924-{stage}-{attempt}',
        route='ue-cnh-route',question='Does CNH add useful corridor alert information over paired scalar readouts?',
        evaluator='research/active/dtr-r0/nearfield/cnh_route_control.py',
        evidence_boundary='Controlled simulation Development; no real hardware performance or safety claim',
        reuse=dict(mode='development',query='CNH RGB native mesh paired corridor source configuration reuse'),
        command=command,
        inputs=inputs or [dict(alias='plan',path=str(ROOT/'plan'),role='configuration',purpose='new-simulation-source-plan')],
        outputs=[dict(alias='result',path=str(result),role='result',required=True)],result_output='result',
        parameters=dict(stage=stage))
    path=ROOT/'run-specs'/f'{stage}-{attempt}.json';write(path,spec);return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage',required=True);parser.add_argument('--status',default='RUNNING')
    parser.add_argument('--note',default='');a=parser.parse_args()
    print(json.dumps(progress(a.stage,a.status,a.note),ensure_ascii=False,indent=2))
