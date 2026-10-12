"""Same-frame comparison of correspondence and gauge transport candidates."""
import json
from pathlib import Path
import numpy as np
from frozen_fusion_eval_v2 import confirm
from fusion_refinement_score_dev_20261013 import metrics

ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts.local/work/fusion-refinement-dev-20261013'
OUT=ROOT/'artifacts.local/work/fusion-refinement-v2-dev-20261013'

def predict(rows):
    return np.stack([confirm([r['rgb'][q] or r['tof_strong'][q] for r in rows]) for q in range(2)],axis=1)

def compare(sources):
    old=json.loads((OLD/'robust_readout.json').read_text())
    combined={key:dict(label=s['label'],session=s['session'],arms={k:s['arms'][k] for k in ('baseline','robust_replace')}) for key,s in old['sessions'].items()}
    for source in sources:
        data=json.loads((OUT/source).read_text())
        assert set(data['sessions'])==set(combined)
        for key,s in data['sessions'].items():
            for name,rows in s['arms'].items():
                if name in ('baseline','robust_replace'):
                    assert rows==combined[key]['arms'][name]
                else:
                    assert name not in combined[key]['arms']
                    combined[key]['arms'][name]=rows
    report=dict(sources=sources,sessions={},aggregate={})
    for key,s in combined.items():
        reference=s['arms']['baseline']
        preds={name:predict(rows) for name,rows in s['arms'].items()}
        arms={}
        for name,rows in s['arms'].items():
            assert len(rows)==len(reference)
            for oldrow,newrow in zip(reference,rows):
                for field in ('i','t_ns','pose_ok','tof_strong','seq','device_ms'):
                    assert oldrow[field]==newrow[field],(key,name,field)
                hist=np.asarray(newrow['occupancy'])
                np.testing.assert_array_equal(hist.sum(axis=(1,2)),newrow['px'])
            o=preds[name];m=metrics(rows,o)
            for ref in ('baseline','robust_replace'):
                b=preds[ref]
                m['vs_'+ref]=dict(query_added=int((o&~b).sum()),query_removed=int((b&~o).sum()),
                                  any_added=int((o.any(1)&~b.any(1)).sum()),any_removed=int((b.any(1)&~o.any(1)).sum()))
            cls=s['label']['class']
            m['required_end']=m['end_H'] if cls=='head' else m['end_any']
            arms[name]=m
            agg=report['aggregate'].setdefault(name,dict(frames=0,rgb_available=0,any_alert=0,query_added=0,query_removed=0,any_added=0,any_removed=0))
            for field in ('rgb_available',):agg[field]+=m[field]
            agg['frames']+=m['n'];agg['any_alert']+=m['any']
            for field in ('query_added','query_removed','any_added','any_removed'):agg[field]+=m['vs_baseline'][field]
        report['sessions'][key]=dict(label=s['label'],arms=arms)
    (OUT/'combined_readout.json').write_text(json.dumps(dict(sessions=combined)),encoding='utf8')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=1),encoding='utf8')
    for name,summary in report['aggregate'].items():
        print(name,summary)
        for key in ('073020','073554','073635','073740','073821'):
            m=report['sessions'][key]['arms'][name]
            print(key,'any',m['any'],'/',m['n'],'end',m['required_end'],'/',m['end_n'],'first',m['first_s'])

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('sources',nargs='+');a=ap.parse_args();compare(a.sources)
