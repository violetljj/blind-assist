"""Promote only after the prewritten contract and sim gates all pass."""
import json
from pathlib import Path
from local_signed_bias_replay import ROOT,OUT,PREV,sha,read

def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def run():
    audit=read(OUT/'independent_audit.json');assert audit['status']=='PASS'
    for name in ('independent_units.json','independent_forward.json','sim_forward_receipt.json','sim_controls_receipt.json'):
        assert read(OUT/name)['status']=='PASS',name
    sim=read(OUT/'sim_evaluation.json');gates=sim['acceptance']
    assert all(gates[k] for k in ('gain_pass','original_default_plus_1pp_pass','timely_small_change_pass','far_clear_small_change_pass'))
    real=read(OUT/'real_receipt.json');assert real['status']=='COMPLETE'
    controls=[r for r in real['tables'] if r['arm']=='main' and r['role']=='deterministic_no_surface']
    assert len(controls)==48 and all(r['ready_frames']==9 and r['strong']==r['light']==0 for r in controls)
    for sid in ('2','3','5','6','8'):
        rows=[r for r in real['tables'] if r['arm']=='main' and r['condition']==sid and r['model']=='S' and (sid!='8' or r['height']=='HEAD')]
        assert any(r['strong']+r['light']>0 for r in rows),sid
    source=PREV/'main_real_manifest.json';manifest=read(source)
    manifest.update(default_scope='Simulation Development; independent real-world confirmation pending',
        input_contract='typed-v4-zero-evidence',contract_module='local_signed_bias_adapter',
        replay_entry='local_signed_bias_default.default_engine',
        real_input_entry='local_signed_bias_adapter.real_adapter',sim_input_entry='local_signed_bias_adapter.sim_adapter',
        raw_score_validity='Respect score_evidence_valid; ungated learned scores retained for diagnostics only',
        parent_manifest=dict(path=source.relative_to(ROOT).as_posix(),sha256=sha(source)),
        decision='Promoted under PLAN c6c77599 gates; 193e4a21 prior real-chain failure preserved',
        adapter_sha256=sha(Path(__file__).with_name('local_signed_bias_adapter.py')),
        evidence_sha256={name:sha(OUT/name) for name in ('v4_input_seal.json','real_receipt.json','sim_evaluation.json','sim_input_closure.json','sim_forward_receipt.json','sim_controls_receipt.json','independent_audit.json','independent_forward.json','independent_units.json')})
    package=OUT/'candidate_manifest.json';write(package,manifest)
    pointer=dict(candidate='no_bin0_DR_S_v4',scope='Simulation Development only',date='2026-10-11',
        manifest_path=package.relative_to(ROOT).as_posix(),manifest_sha256=sha(package),
        contract_path='research/active/hardware-bringup/host/local_signed_bias_adapter.py',contract_sha256=manifest['adapter_sha256'],
        loader='research/active/hardware-bringup/host/local_signed_bias_default.py',
        report='research/active/hardware-bringup/LOCAL_SIGNED_BIAS_CONTRACT_DEV_20261011.md',
        predecessor='9e150dd8 old S baseline; 193e4a21 no_bin0_DR_S now succeeds through v4 contract')
    write(ROOT/'research/active/dtr-r0/nearfield/CNH_SIM_DEFAULT_CANDIDATE.json',pointer)
    write(OUT/'promotion_receipt.json',dict(status='PROMOTED_SIM_DEVELOPMENT_DEFAULT',all_gates_pass=True,
        weights_changed=False,cuts_changed=False,training=0,pilot_used_for_selection=False,
        manifest_sha256=sha(package),pointer_sha256=sha(ROOT/'research/active/dtr-r0/nearfield/CNH_SIM_DEFAULT_CANDIDATE.json'),
        simulation_acceptance=gates,independent_audit_sha256=sha(OUT/'independent_audit.json')))
    from local_signed_bias_default import default_manifest
    assert default_manifest()==package
    print('PROMOTED_SIM_DEVELOPMENT_DEFAULT; static default loader hash check PASS',flush=True)

if __name__=='__main__':run()
