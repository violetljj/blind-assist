"""Focused implementation/evidence checks, not generalization evidence."""
import json
from pathlib import Path
import numpy as np
from fusion_anchor_candidates_dev_20261013 import numerical_checks, robust_numerical_checks
from fusion_temporal_candidates_dev_20261013 import numerical_checks as temporal_checks, run_candidates
from frozen_fusion_eval_v2 import confirm

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/fusion-refinement-dev-20261013'

def main():
    result=dict(anchor=numerical_checks(),robust=robust_numerical_checks(),temporal=temporal_checks())
    data=json.loads((OUT/'robust_readout.json').read_text())
    cache=json.loads((OUT/'cache_manifest.json').read_text())
    parity=json.loads((OUT/'baseline_parity.json').read_text())
    assert not parity['mismatches']
    total=0;histograms=0;arm_summaries={}
    for key,s in data['sessions'].items():
        baseline=s['arms']['baseline']; total+=len(baseline)
        legacy=run_candidates(baseline)['baseline_2of3']
        for q in range(2):
            exact=confirm([r['rgb'][q] or r['tof_strong'][q] for r in baseline])
            np.testing.assert_array_equal(legacy[:,q],exact)
        for name,rows in s['arms'].items():
            assert [r['i'] for r in rows]==[r['i'] for r in baseline]
            counts=arm_summaries.setdefault(name,dict(available=0,fresh=0,reused=0,robust_recovered=0))
            for r in rows:
                occ=np.asarray(r['occupancy'])
                assert occ.shape==(2,6,8) and (occ>=0).all()
                np.testing.assert_array_equal(occ.sum(axis=(1,2)),r['px'])
                assert r['rgb']==[p>=150 and r['available'] for p in r['px']]
                counts['available']+=int(r['available'])
                counts['reused']+=int(r['anchor']=='reused')
                counts['fresh']+=int(r['anchor'] in ('tof','floor','robust_tof'))
                counts['robust_recovered']+=int(r['anchor']=='robust_tof')
                histograms+=2
    assert total==sum(len(s['frames']) for s in cache['sessions'].values())
    result.update(sessions=len(data['sessions']),frames=total,legacy_query_decisions=total*2,
                  histogram_checks=histograms,anchor_coverage=arm_summaries,cache_bytes=cache['cache_bytes'],
                  cache_wall_s=cache['cache_wall_s'],readout_wall_s=data['wall_s'],robust_wall_s=data['robust_wall_s'])
    (OUT/'verification.json').write_text(json.dumps(result,indent=1),encoding='utf8')
    print(json.dumps(result,indent=1))

if __name__=='__main__':main()
