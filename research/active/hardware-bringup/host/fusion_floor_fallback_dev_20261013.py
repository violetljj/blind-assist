"""Successful floor rescue only; preserve robust fallback when floor unavailable."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts.local/work/fusion-refinement-dev-20261013'
OUT=ROOT/'artifacts.local/work/fusion-refinement-v2-dev-20261013'

def main():
    old=json.loads((OLD/'robust_readout.json').read_text())
    result=dict(sessions={},rule='quality-approved ToF else available floor else robust_replace')
    changes={}
    for key,s in old['sessions'].items():
        arms={k:s['arms'][k] for k in ('baseline','robust_replace')}
        rows=[];count=0
        for robust,floor in zip(s['arms']['robust_replace'],s['arms']['quality_floor']):
            if floor['anchor']=='floor' and floor['available']:
                row=copy.deepcopy(floor);row['anchor']='floor_rescue';count+=1
            else:row=copy.deepcopy(robust)
            row['floor_rescue_applied']=floor['anchor']=='floor' and floor['available']
            assert row['available']==robust['available']
            rows.append(row)
        arms['floor_rescue']=rows;changes[key]=count
        result['sessions'][key]=dict(session=s['session'],label=s['label'],arms=arms)
    result['floor_frames']=changes
    (OUT/'floor_rescue_readout.json').write_text(json.dumps(result),encoding='utf8')
    print(changes)

if __name__=='__main__':main()
