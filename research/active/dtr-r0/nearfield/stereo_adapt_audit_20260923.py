"""Read-only post-seal audit; group reporting is descriptive, never IID testing."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4];ART=ROOT/'artifacts.local'
MODELS=ART/'evidence/ba-stereo-adapt-20260923-models'
PRED=ART/'evidence/ba-stereo-adapt-20260923-predictions'
EVAL=ART/'evidence/ba-stereo-adapt-20260923-evaluation'
PREP=ART/'evidence/ba-stereo-adapt-20260923-prepared'
OLD=ART/'evidence/ba-vpp-geometry-20260923-frontend-v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def check_file(folder,receipt,name):
    assert sha(folder/name)==receipt['hashes'][name],(folder,name)
def correct(row):return row['truth'] is not None and row['pred'] is not None and abs(row['pred']-row['truth'])<=.05


def negative_query_far_support(truth_rows,pixel_rows):
    """Default-width no-contact geometry negatives, not all-frame pixel totals."""
    truth={}
    for row in truth_rows:
        key=(row['cohort'],row['panel'],row['id'])
        assert key not in truth
        truth[key]=row['truth']['distance']
    counts=defaultdict(Counter);seen=set()
    for row in pixel_rows:
        key=(row['cohort'],row['panel'],row['id'])
        unique=key+(row['arm'],row['part'])
        assert unique not in seen
        seen.add(unique)
        # Frozen width index 1: BODY .56 m / HEAD .36 m; horizon 4 m.
        part=row['part'];assert part in ('BODY','HEAD')
        if truth[key][part][1] is not None:continue
        count=counts[(row['cohort'],row['arm'],part)]
        pixels=row['far_native_estimated_query'];assert isinstance(pixels,int) and pixels>=0
        count['negative_queries']+=1
        count['far_native_estimated_query_pixels']+=pixels
        count['negative_queries_with_far_support']+=pixels>0
    totals=defaultdict(Counter)
    for (cohort,arm,part),count in counts.items():totals[(cohort,arm)].update(count)
    return dict(definition='Default width BODY .56m / HEAD .36m, camera axial contact horizon .5..4m; native geometry truth contact is None. Counts refer to stereo depth support, not ToF union or final alerts.',
        totals=[dict(cohort=k[0],arm=k[1],**v) for k,v in sorted(totals.items())],
        rows=[dict(cohort=k[0],arm=k[1],part=k[2],**v) for k,v in sorted(counts.items())])


def grouped_pairs(rows,layout_by_id):
    keyed=defaultdict(dict)
    for r in rows:
        key=(r['cohort'],r['panel'],r['id'],r['part'],r.get('width'))
        assert r['arm'] not in keyed[key]
        keyed[key][r['arm']]=r
    results={}
    for first,second in [('baseline','ordinary'),('baseline','balanced'),('ordinary','balanced')]:
        for suffix in ('','_union'):
            groups=defaultdict(Counter)
            for key,arms in keyed.items():
                a,b=arms[first+suffix],arms[second+suffix]
                assert a['truth']==b['truth']
                if key[0]=='new_eval':group=layout_by_id[a['id']]
                else:group=a['panel']+'/'+a['episode']
                count=groups[(a['cohort'],a['family'],group)]
                count['queries']+=1
                count['truth_contact']+=a['truth'] is not None
                count['corrected_5cm']+=not correct(a) and correct(b)
                count['regressed_5cm']+=correct(a) and not correct(b)
                count['recovered_missing']+=a['truth'] is not None and a['pred'] is None and b['pred'] is not None
                count['lost_contact']+=a['truth'] is not None and a['pred'] is not None and b['pred'] is None
                count['removed_false_contact']+=a['truth'] is None and a['pred'] is not None and b['pred'] is None
                count['added_false_contact']+=a['truth'] is None and a['pred'] is None and b['pred'] is not None
            detail=[dict(cohort=k[0],family=k[1],group=k[2],**v,
                net_correct_5cm=v['corrected_5cm']-v['regressed_5cm']) for k,v in sorted(groups.items())]
            families={}
            for cohort,family in sorted({(d['cohort'],d['family']) for d in detail}):
                selected=[d for d in detail if d['cohort']==cohort and d['family']==family]
                families[cohort+'/'+family]=dict(groups=len(selected),
                    groups_net_improved=sum(d['net_correct_5cm']>0 for d in selected),
                    groups_net_regressed=sum(d['net_correct_5cm']<0 for d in selected),
                    groups_net_tied=sum(d['net_correct_5cm']==0 for d in selected),
                    corrected_5cm=sum(d['corrected_5cm'] for d in selected),regressed_5cm=sum(d['regressed_5cm'] for d in selected))
            results[first+'_to_'+second+suffix]=dict(groups=detail,families=families)
    return results


def audit(result,models=MODELS,predictions=PRED,evaluation=EVAL,prepared=PREP):
    journal=read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])
    assert journal['state']=='running' and journal['reuse_preflight']['status']=='PASS'
    assert not result.exists(),'Preserve existing audit'
    import torch
    model_checks={};orders={};before_states={};training_receipts={}
    for arm in ('ordinary','balanced'):
        folder=models/arm;receipt=read(folder/'receipt.json');training_receipts[arm]=receipt
        assert receipt['status']=='PASS' and receipt['stage']=='train' and receipt['arm']==arm
        assert receipt['steps']==receipt['frames']==192 and receipt['iterations']==8
        for filename in ('before-hashes.json','after-hashes.json','sample-order.json','steps.jsonl','final-delta.pt'):check_file(folder,receipt,filename)
        before,after=read(folder/'before-hashes.json'),read(folder/'after-hashes.json')
        before_states[arm]=before
        assert before['frozen']==after['frozen'] and before['buffers']==after['buffers']
        order=read(folder/'sample-order.json');orders[arm]=order
        assert [r['step'] for r in order]==list(range(192))
        assert len({r['id'] for r in order})==192
        layout_counts=Counter(r['layout_id'] for r in order)
        assert len(layout_counts)==48 and set(layout_counts.values())=={4}
        steps=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(steps)==192
        assert [(r['step'],r['id'],r['layout_id']) for r in steps]==[(r['step'],r['id'],r['layout_id']) for r in order]
        assert all(np.isfinite(r['gradient_norm']) and all(np.isfinite(v) and v>0 for v in r['gradient_max_by_module'].values()) for r in steps)
        delta=torch.load(folder/'final-delta.pt',map_location='cpu',weights_only=True)
        assert tuple(delta['selected_modules'])==('update_block','spx_2_gru','spx_gru')
        assert set(delta['state'])==set(after['selected'])
        for name,value in delta['state'].items():
            assert hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()==after['selected'][name]
        changed=[name for name in before['selected'] if before['selected'][name]!=after['selected'][name]]
        assert set(changed)==set(receipt['changed_selected_parameters'])
        model_checks[arm]=dict(steps=192,unique_layouts=48,poses_per_layout=4,
            frozen_parameters_and_buffers_unchanged=True,delta_exactly_matches_final_selected_weights=True,
            delta_sha256=sha(folder/'final-delta.pt'),base_checkpoint_sha256=delta['base_checkpoint_sha256'],
            finite_gradient_steps=192,loss_values_not_compared=True)
        del delta
    assert orders['ordinary']==orders['balanced']
    assert before_states['ordinary']==before_states['balanced']
    assert model_checks['ordinary']['base_checkpoint_sha256']==model_checks['balanced']['base_checkpoint_sha256']
    assert training_receipts['ordinary']['optimizer']==training_receipts['balanced']['optimizer']
    assert training_receipts['ordinary']['input_manifest_sha256']==training_receipts['balanced']['input_manifest_sha256']
    inference={};receipts={}
    for arm,total in [('baseline',96),('ordinary',672),('balanced',672)]:
        folder=predictions/arm;receipt=read(folder/'receipt.json');receipts[arm]=receipt
        assert receipt['status']=='PASS' and receipt['stage']=='infer' and receipt['arm']==arm and receipt['frames']==total
        assert receipt['iterations']==32 and receipt['target_access']=='NONE_PUBLIC_INPUTS_ONLY'
        check_file(folder,receipt,'hints.json');check_file(folder,receipt,'manifest.json')
        hints=read(folder/'hints.json')['frames'];assert len(hints)==total
        lookup={(r['panel'],r['id']):r for r in hints};assert len(lookup)==total
        inference[arm]=lookup
        if arm!='baseline':assert receipt['updates_sha256']==model_checks[arm]['delta_sha256']
    assert len({r['input_manifest_sha256'] for r in receipts.values()})==1
    fields=('seed','total_frame_hints','seeds','changed_left_pixels','changed_right_pixels','left_pixel_sha256','right_pixel_sha256')
    new_keys=set(inference['baseline']);assert len(new_keys)==96 and all(k[0]=='new_eval' for k in new_keys)
    for key in new_keys:
        a=inference['baseline'][key]
        assert a['seed_panel']=='adapt'
        for arm in ('ordinary','balanced'):
            b=inference[arm][key];assert b['seed_panel']=='adapt'
            assert all(a[field]==b[field] for field in fields),(arm,key)
    old_receipt=read(OLD/'receipt.json');check_file(OLD,old_receipt,'hints.json')
    old={(r['panel'],r['id']):r for r in read(OLD/'hints.json')['frames']}
    assert len(old)==576
    for arm in ('ordinary','balanced'):
        assert set(inference[arm])==new_keys|set(old)
        for key,a in old.items():
            b=inference[arm][key];assert b['seed_panel']==key[0]
            assert all(a[field]==b[field] for field in fields),(arm,key)
    ev=read(evaluation/'receipt.json');assert ev['status']=='PASS'
    for name in ('summary.json','distance-queries.json','width-queries.json','prediction-seal.json','truth.json','pixel-queries.json'):check_file(evaluation,ev,name)
    public=read(prepared/'eval_observations/examples.json')
    prep=read(prepared/'receipt.json');check_file(prepared,prep,'eval_observations/examples.json')
    layouts={r['id']:r['layout_id'] for r in public['frames']}
    assert len(layouts)==96 and len(set(layouts.values()))==24
    output=dict(status='PASS',same_initial_weights=True,same192_example_order=True,training=model_checks,
        projected_inputs=dict(new_eval_96_all_three_arms_equal=True,historical576_both_adapters_equal_original_vpp=True),
        grouped_distance=grouped_pairs(read(evaluation/'distance-queries.json'),layouts),
        grouped_width=grouped_pairs(read(evaluation/'width-queries.json'),layouts),
        negative_query_far_support=negative_query_far_support(read(evaluation/'truth.json'),read(evaluation/'pixel-queries.json')),
        evidence_boundary='Descriptive paired results; new layouts are groups, historical episodes are groups; widths/poses are not independent replicates; no p-values or training-loss objective ranking',
        inputs=dict(model_receipts={a:sha(models/a/'receipt.json') for a in model_checks},
            prediction_receipts={a:sha(predictions/a/'receipt.json') for a in receipts},evaluation_receipt=sha(evaluation/'receipt.json')),
        audit_code_sha256=sha(__file__))
    result.parent.mkdir(parents=True,exist_ok=True)
    with result.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print('STEREO_ADAPT_AUDIT_PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
    for name,default in [('models',MODELS),('predictions',PRED),('evaluation',EVAL),('prepared',PREP)]:p.add_argument('--'+name,type=Path,default=default)
    a=p.parse_args();audit(a.result,a.models,a.predictions,a.evaluation,a.prepared)
