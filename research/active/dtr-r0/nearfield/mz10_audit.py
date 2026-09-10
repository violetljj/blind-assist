"""Scalar verification of the fixed composition; no fits or alternate routes."""
import argparse
from pathlib import Path
import numpy as np
from mz5_ensemble_readout import read,write,load_npz,sha
from mz10_availability import compose


def main(run):
    d=load_npz(run/'predictions.npz');r=read(run/'result.json');groups=0;selections=0
    for cohort in ['DEV','clean','stress']:
        t=d[cohort+'/truth'];s=d[cohort+'/available'];b=d[cohort+'/baseline'];c=d[cohort+'/composite'];z=d[cohort+'/source']
        for i in range(len(t)):
            for q in range(4):
                assert c[i,q]==(z[i,q] if s[i,q] else b[i,q]);selections+=1
        for arm,expected in r[cohort]['metrics'].items():
            pred=d[cohort+'/'+arm];tp=[0]*4;fp=[0]*4;positives=[0]*4;exact=0
            for i in range(len(t)):
                count=0
                for q in range(4):
                    v=bool(pred[i,q]>=0);truth=bool(t[i,q]);tp[q]+=v and truth;fp[q]+=v and not truth;positives[q]+=truth;count+=v==truth
                exact+=count==4
            assert expected==dict(exact=exact,tp=tp,fp=fp,positives=positives);groups+=1
        assert np.array_equal(c[~s],b[~s])
    for clip,arms in r['clips'].items():
        ids=np.flatnonzero(d['clip']==clip)
        for arm,expected in arms.items():
            t=d['clean/truth'];z=d['clean/'+arm]
            exact=sum(all(bool(z[i,q]>=0)==bool(t[i,q]) for q in range(4)) for i in ids)
            assert exact==expected['exact'];groups+=1
    # Adversarial scores cannot change which branch is selected.
    b=np.array([[100.,-100.,2.,-2.]]);z=-b;s=np.array([[False,True,False,True]])
    np.testing.assert_array_equal(compose(b,z,s),[[100.,100.,2.,2.]])
    previous=load_npz(run.parent/'run-v1/predictions.npz')
    for cohort in ['DEV','clean','stress']:
        for arm in ['baseline','adapted','source','composite']:
            np.testing.assert_array_equal(d[cohort+'/'+arm]>=0,previous[cohort+'/'+arm]>=0)
    out=dict(status='PASS',scalar_metric_groups=groups,scalar_branch_selections=selections,
        recovery_note='run-v1 reused clean frozen logits for stress; run-v2 uses actual frozen stress logits. All signs and outcomes identical; no fit/threshold/routing change.',
        predicate='Only observed geometric support selects source, never truth or logit magnitude',
        script_sha256=sha(Path(__file__)))
    write(run/'audit.json',out);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
