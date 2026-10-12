"""Resume one observed JSON scalar failure using saved hold grades; no forward.

The original manifest and failure stay immutable. The only evaluator repair is
scene=n -> scene=int(n). Existing grade archives and ledgers must stay identical.
"""
import json
from pathlib import Path
import time
import numpy as np
import cnh_counterfactual_common_dev as C
import cnh_cost_v2_metrics_20261010 as V
import cnh_cost_v2_holdout_dev_20261010 as R


def main():
    began=time.monotonic(); out=R.OUT
    manifest=C.read(out/'execution_manifest.json')
    metric_path=Path(V.__file__)
    original=C.sha(metric_path)
    for name,digest in {**manifest['inherited'],**manifest['new_sources']}.items():
        if Path(name) != metric_path and C.sha(name)!=digest:
            raise ValueError('Unbound source drift: '+name)
    repaired=metric_path.read_text(encoding='utf8')
    old=repaired.replace('scene_counts=[dict(scene=int(n), denominator=',
                         'scene_counts=[dict(scene=n, denominator=',1)
    import hashlib
    # Git working text was CRLF before the one-line apply_patch normalization.
    candidates=[old.encode(),old.replace('\n','\r\n').encode()]
    if manifest['new_sources'][str(metric_path)] not in [hashlib.sha256(x).hexdigest() for x in candidates]:
        raise ValueError('Repair exceeds the one scalar conversion')
    retained=['hold_grades_notifications.npz','event_ledger.csv','notification_ledger.csv']
    hashes={n:C.sha(out/n) for n in retained}
    binding=dict(reason='numpy int64 scene ID JSON failure',
        original_manifest_sha256=C.sha(out/'execution_manifest.json'),
        original_metric_sha256=manifest['new_sources'][str(metric_path)],
        repaired_metric_path=str(metric_path),repaired_metric_sha256=original,
        recovery_source_sha256=C.sha(Path(__file__)),retained_outputs_sha256=hashes,
        permitted_change='contact_summary scene ID int(n); identical numeric evaluation',
        inference=0,calibration_reselection=0)
    binding_path=out/'serialization_repair_binding.json'
    if not binding_path.exists():R.save(binding_path,binding)
    else:
        prior=C.read(binding_path)
        for key in ('original_manifest_sha256','original_metric_sha256','repaired_metric_sha256','retained_outputs_sha256'):
            assert binding[key]==prior[key]
    with np.load(out/'hold_grades_notifications.npz',allow_pickle=False) as a:
        saved={n:a[n].copy() for n in a.files}
    keys=list(saved['keys']); grades=saved['grades']
    scores=dict(m3=grades[keys.index('fixed/m3')],old5=grades[keys.index('fixed/old5')],
        both=np.asarray([grades[keys.index(f'{seed}/both')] for seed in V.G.SEEDS]),
        margin=saved['margin'],joint_scores=saved['joint_scores'])
    def cache_score(root,split):
        assert root==out and split=='hold'
        return scores
    original_score=V.F.score_split; original_save=np.savez_compressed
    def retain_archive(path,**arrays):
        assert Path(path)==out/'hold_grades_notifications.npz'
        assert set(arrays)==set(saved)
        for key,value in arrays.items():np.testing.assert_array_equal(value,saved[key])
    def check():
        if time.monotonic()-began>=100:raise TimeoutError('Serialization recovery budget')
    try:
        V.F.score_split=cache_score;np.savez_compressed=retain_archive
        if not (out/'metrics.json').exists():
            detail=V.evaluate(out,C.read(out/'scene_rows.json'),check)
        else:
            # Previous recovery completed metrics but its receipt construction failed.
            detail=dict(outputs_sha256={n:C.sha(out/n) for n in retained+['metrics.json']},
                        source_sha256=C.sha(metric_path),training=0,protected_access=0)
        for name,digest in hashes.items():assert C.sha(out/name)==digest,name
        R.save(out/'evaluation_receipt.json',{**detail,'status':'COMPLETE',
            'seconds':time.monotonic()-began,'hgb_forward_seconds':0,
            'recovery':'saved hold grades, margins, joint scores; no new forward',
            'recovery_source_sha256':C.sha(Path(__file__)),
            'repair_binding_sha256':C.sha(binding_path)})
    finally:
        V.F.score_split=original_score;np.savez_compressed=original_save
    print('RECOVERED_SERIALIZATION',round(time.monotonic()-began,3))


if __name__=='__main__':main()
