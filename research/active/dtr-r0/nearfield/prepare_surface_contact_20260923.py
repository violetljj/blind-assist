"""Scoped governance and launch seal for the frozen-depth contact experiment."""
import json,sys
from pathlib import Path
import run_surface_contact_20260923 as run
import vpp_geometry_prepare_20260923 as reuse

ROOT,ART,BASE=run.ROOT,run.ART,run.BASE
PROTOCOL='research/active/dtr-r0/nearfield/SURFACE_CONTACT_PROTOCOL_20260923.md'

def setup():
    BASE.mkdir(parents=True,exist_ok=True)
    contracts=[]
    for locator,status,scopes in [
        ('evidence/ba-surface-contact-20260923','source_material',[('plan','configuration')]),
        ('evidence/ba-surface-contact-20260923-predictions-v1','development_consumed',[('.','evaluator')]),
        ('evidence/ba-surface-contact-20260923-evaluation-v1','development_consumed',[('.','evaluator')])]:
        contracts.append(dict(locator=locator,source_family='ue-historical-stereo-mz101-mz102',evidence_status=status,
            evidence=PROTOCOL,allowed_inputs=[dict(relative_path=p,role=r) for p,r in scopes]))
    policy=ROOT/'data/ue-reuse-policy.json';old=policy.read_text(encoding='utf-8');existing=json.loads(old)
    assert not {c['locator'] for c in contracts}&{c['locator'] for c in existing['contracts']}
    marker='"contracts": ['
    insertion='\n'+',\n'.join('    '+json.dumps(c,indent=2).replace('\n','\n    ') for c in contracts)+','
    new=old.replace(marker,marker+insertion,1);json.loads(new);policy.write_bytes(new.encode())
    run.write(BASE/'own-policy-contracts.json',contracts)
    near=ROOT/'research/active/dtr-r0/nearfield'
    files=[ROOT/PROTOCOL,Path(run.__file__),Path(__file__),near/'surface_contact_core.py',near/'surface_contact_oracle.py']
    run.write(BASE/'plan/launch-seal.json',dict(status='FROZEN_BEFORE_COHORT_BOUNDARY_RESULTS',hashes={str(p):run.sha(p) for p in files}))
    def inp(alias,path,role):return dict(alias=alias,path=str(path),role=role,purpose='frozen-depth-camera-contact-boundary')
    plan=inp('plan',BASE/'plan','configuration')
    public=[inp('tof',run.PUBLIC/'tof','observation'),inp('tof_manifest',run.PUBLIC/'tof-inputs.json','observation'),inp('rgb_manifest',run.PUBLIC/'rgb-inputs.json','observation')]
    models=[inp(m,p,'evaluator') for m,p in run.MODELS.items()]
    for stage in ('canary','predict','evaluate','audit'):
        if stage in ('canary','predict'):
            inputs=[plan,*public,*models];dest=(BASE/'canary' if stage=='canary' else run.PRED)/'receipt.json'
        else:
            inputs=[plan,inp('predictions',run.PRED,'evaluator'),*[inp(k,v,'evaluator') for k,v in run.CAPTURES.items()],
                inp('previous_geometry',ART/'evidence/ba-vpp-geometry-20260923-evaluation-v1','evaluator')]
            dest=run.EVAL/'summary.json'
            if stage=='audit':
                inputs.extend([inp('evaluation',run.EVAL,'evaluator'),*models,*public]);dest=ART/'evidence/ba-surface-contact-20260923-audit-v1/result.json'
        command=[sys.executable,str(Path(run.__file__)),stage]
        if stage=='audit':command=[sys.executable,str(near/'surface_contact_audit_20260923.py')]
        spec=dict(schema='blindassist-asset-run-v1',id='surface-contact-20260923-'+stage,route='ue-surface-contact',
            question='Can bounded local surfaces recover camera-forward contact boundaries from frozen depths?',evaluator=PROTOCOL,
            evidence_boundary='Consumed historical rendered Development; new camera-aligned labels, not current-A or hardware',
            reuse=dict(mode='diagnostic',query='MZ101 MZ102 frozen depth bounded local surface contact boundary'),
            inputs=inputs,outputs=[dict(alias='result',path=str(dest),role='result',required=True)],result_output='result',command=command)
        run.write(BASE/(stage+'-run-spec.json'),spec)
    registrations=[]
    for p in [BASE/'plan',run.PUBLIC, *run.MODELS.values(),*run.CAPTURES.values()]:registrations.append(reuse.register(p,'source_material' if p==BASE/'plan' else 'development_consumed'))
    run.write(BASE/'registration.json',registrations)
    print('SURFACE_CONTACT_PREPARED')

if __name__=='__main__':setup()
