"""Independent stdlib saved-row audit, no production metric imports."""
import json, os, sys, hashlib
from pathlib import Path
from collections import defaultdict
REPO=Path(__file__).resolve().parents[4]
ROOT=REPO/'artifacts.local/evidence/ba-local-rescue-audit-20260923'
EVIDENCE=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def runs(clip,pred):
    groups=[]
    for row in clip:
        if pred(row):
            if not groups or groups[-1][-1]['frame_in_clip']+1!=row['frame_in_clip']:groups.append([])
            groups[-1].append(row)
    return groups
def measure(rows,arm):
    counts={k:0 for k in ('TP','FP','FN','TN','prediction_unknown','abstained_negative','abstained_positive')}
    clips=defaultdict(list)
    for r in rows:
        clips[r['clip_id']].append(r)
        if r['truth'] is None:continue
        p=r['predictions'][arm];y=r['truth']
        counts['prediction_unknown']+=p['unknown']
        if p['alert']:counts['TP' if y else 'FP']+=1
        elif y:counts['FN']+=1
        elif not p['unknown']:counts['TN']+=1
        if p['unknown'] and not p['alert']:counts['abstained_positive' if y else 'abstained_negative']+=1
    events=[];segments=[]
    for cid,clip in sorted(clips.items()):
        clip.sort(key=lambda r:r['frame_in_clip'])
        assert [r['frame_in_clip'] for r in clip]==list(range(len(clip)))
        for group in runs(clip,lambda r:r['truth'] is True):
            first=next((r['time_s'] for r in group if r['predictions'][arm]['alert']),None)
            events.append((cid,group[0]['frame_in_clip'],group[-1]['frame_in_clip'],first))
        for group in runs(clip,lambda r:r['truth'] is False and r['predictions'][arm]['alert']):
            segments.append((cid,group[0]['frame_in_clip'],group[-1]['frame_in_clip']))
    return dict(counts=counts,events=events,segments=segments,detected_events=sum(e[3] is not None for e in events))
def execute():
    assert read(Path(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL']))['state']=='running'
    result={};checks=0
    for stem in ('ba-local-rescue-20260923-run','ba-local-rescue-gate-20260923-run'):
        path=EVIDENCE/stem
        for f,h in read(path/'output-seal.json')['files'].items():assert sha(path/f)==h;checks+=1
        for cohort in ('transfer','stability'):
            rows=read(path/(cohort+'-rows.json'));report=read(path/(cohort+'-report.json'))
            measured={}
            for arm,saved in report['metrics']['arms'].items():
                m=measure(rows,arm)
                for k,v in m['counts'].items():assert saved['frames']['all_known'][k]==v,(stem,cohort,arm,k);checks+=1
                assert m['events']==[(e['clip_id'],e['start_frame'],e['end_frame'],e['first_alert_time_s']) for e in saved['events']]
                assert m['segments']==[(e['clip_id'],e['start_frame'],e['end_frame']) for e in saved['false_alert_segments']]
                assert m['detected_events']==saved['detected_events'];checks+=3
                measured[arm]=m
            inc=[r for r in rows if r['predictions']['local']['alert'] and not r['predictions']['A_current']['alert']]
            summary=dict(incremental_TP=sum(r['truth'] is True for r in inc),incremental_FP=sum(r['truth'] is False for r in inc))
            if 'ambiguity_gate' in measured:
                gate=read(path/'gate.json');features=read(EVIDENCE/'ba-local-rescue-20260923-run'/(cohort+'-public-decisions.json'))['features']
                for r,f in zip(rows,features,strict=True):
                    assert r['predictions']['ambiguity_gate']['alert']==bool(r['predictions']['A_current']['alert'] or (r['predictions']['local']['alert'] and f['possible_fraction']<=gate['cutoff']));checks+=1
                    assert r['predictions']['ambiguity_gate']['unknown']==r['predictions']['A_current']['unknown']
                summary.update(retained_TP=sum(r['truth'] is True and r['predictions']['ambiguity_gate']['alert'] for r in inc),removed_FP=sum(r['truth'] is False and not r['predictions']['ambiguity_gate']['alert'] for r in inc))
                pairs=list(zip(measured['local']['events'],measured['ambiguity_gate']['events'],strict=True))
                summary.update(lost_events=sum(a[3] is not None and b[3] is None for a,b in pairs),delayed_events=sum(a[3] is not None and b[3] is not None and b[3]>a[3]+1e-9 for a,b in pairs))
                for k,v in summary.items():assert report['summary'][k]==v;checks+=1
            else:
                saved=read(path/(cohort+'-incremental.json'))
                assert [r['id'] for r in inc if r['truth'] is not None]==[r['id'] for r in saved];checks+=1
            result[stem+'/'+cohort]=dict(frames=len(rows),summary=summary,arms=measured)
    transfer=read(EVIDENCE/'ba-local-rescue-20260923-run/transfer-incremental.json')
    positive=[r['features']['possible_fraction'] for r in transfer if r['truth'] is True]
    negative=[r['features']['possible_fraction'] for r in transfer if r['truth'] is False]
    gate=read(EVIDENCE/'ba-local-rescue-gate-20260923-run/gate.json')
    assert max(positive)<min(negative)
    assert gate['cutoff']==(max(positive)+min(negative))/2
    result.update(status='PASS',checks=checks,cutoff=gate['cutoff'],transfer_selection=[len(positive),len(negative)],scope='Consumed Development; feature selected after both cohorts; no fresh validation',code_review=['Public feature slot 939 verified against extract layout: 910+15+14.','Causal controls use only current public input and immediately previous same-clip probability.','All A alerts and UNKNOWN flags structurally retained.','Timing measures sampled event onset, not physical/device latency.','Gate may retain 80 percent of LOCAL-only TP; full FN costs must remain visible.'])
    write(ROOT/'run/result.json',result)
    compact={k:{'frames':v['frames'],'summary':v['summary'],'arms':{a:dict(**m['counts'],events=m['detected_events'],total_events=len(m['events']),FP_segments=len(m['segments'])) for a,m in v['arms'].items()}} for k,v in result.items() if isinstance(v,dict) and 'arms' in v}
    write(ROOT/'run/compact.json',compact)
    print(json.dumps(dict(status='PASS',checks=checks,cutoff=gate['cutoff'],cohorts=compact)))
if __name__=='__main__':execute()

