"""Fixed HEAD appearance pairs and evaluator-only native diagnostic cache."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def generate(source, output):
    specs = {r: read(source/f'spec-{r}-v1.json') for r in ('train','eval')}
    result = copy.deepcopy(specs['train'])
    result['cases'] = []
    base = '/Game/Building/SF/B/Material/MI_Bldg_PaintedMetal_'
    swap = {base+'Grey':base+'Blue', base+'Blue':base+'Grey'}
    units = {}
    for role, original in specs.items():
        ids = list(dict.fromkeys(c['group_id'] for c in original['cases']))[:8]
        units[role] = ids
        assert len(ids) == 8
        families = [next(c['condition']['family'] for c in original['cases'] if c['group_id']==g) for g in ids]
        assert sorted(Counter(families).values()) == [2,2,2,2]
        for group in ids:
            for relation in ('CLEAR','HEAD_ONLY'):
                matches = [c for c in original['cases'] if c['group_id']==group and c['variant_id']==relation]
                assert len(matches)==1
                parent = matches[0]
                assert parent['geometric_contact']['relation']==relation
                for appearance in ('A','L','M'):
                    case = copy.deepcopy(parent)
                    case.update(name=f'head-p1-{group}-{relation}-{appearance}', parent_case_name=parent['name'],
                        source_partition=role, source_exposure='exposedTRAIN' if role=='train' else 'consumedEVAL',
                        appearance=appearance, relation=relation)
                    case['condition']['condition_id']=case['name']
                    if appearance=='L':
                        case['sun_intensity_scale']=float(parent.get('sun_intensity_scale',1.))*.55
                        case['skylight_intensity_scale']=float(parent.get('skylight_intensity_scale',1.))*.70
                    elif appearance=='M':
                        changed=0
                        for obj in case['objects']:
                            if obj.get('material_asset') in swap:
                                obj['material_asset']=swap[obj['material_asset']]
                                changed+=1
                        assert changed>0
                    result['cases'].append(case)
    result.update(schema='head-paired-appearance-v1', source_role='DIAGNOSTIC_ONLY',
        purpose='HEAD_P1_PAIRED_APPEARANCE_NO_MODEL_BASED_SELECTION',
        source_inputs={r:dict(path=str(source/f'spec-{r}-v1.json'),sha256=sha(source/f'spec-{r}-v1.json')) for r in specs},
        generator_sha256=sha(Path(__file__)),
        appearance_contract=dict(A='original',L=dict(sun_multiplier=.55,skylight_multiplier=.70),M=swap),
        suite_contract=dict(units=16,frames=96,relations=['CLEAR','HEAD_ONLY'],appearances=['A','L','M']),
        scope='Consumed Development geometry; native support and UNKNOWN equality required within appearance pairs; no new-world confirmation')
    assert len(result['cases'])==96
    write(output/'spec-all-v1.json',result)
    canary=copy.deepcopy(result)
    canary['cases']=[c for c in canary['cases'] if c['group_id'] in (units['train'][0],units['eval'][1])]
    canary['purpose']='NATIVE_APPEARANCE_CANARY_ONLY'
    canary['suite_contract']=dict(units=2,frames=12,relations=['CLEAR','HEAD_ONLY'],appearances=['A','L','M'])
    write(output/'spec-canary-v1.json',canary)
    print(json.dumps(dict(status='PASS',frames=96,canary_frames=12,units=units,appearance_contract=result['appearance_contract'])))


def build(capture, output):
    import numpy as np
    from PIL import Image
    import torch
    spec=read(capture/'source/spec.json')
    truth=read(capture/'world-verification.json')
    assert truth['status']==read(capture/'completion.json')['status']=='PASS'
    release=read(capture/'process-release.json')
    assert release['released'] and not release['survivors']
    assert read(capture/'source-integrity.json')['unchanged']
    assert truth['mask_order']==['BODY','HEAD'] and truth['query_range_m']==3.
    assert len(spec['cases'])==len(truth['rows']) and torch.cuda.is_available()
    rgb,near,support,records,native_masks=[],[],[],[],[]
    output.mkdir(parents=True,exist_ok=False)
    native_rgb=[]
    for i,(case,row) in enumerate(zip(spec['cases'],truth['rows'])):
        assert case['name']==row['name']
        ip=capture/f'model/sample/{i:04d}.png'; mp=capture/row['mask_path']
        assert sha(ip)==row['rgb_sha256'] and sha(mp)==row['mask_sha256']
        image=Image.open(ip).convert('RGB')
        assert image.size==(640,360)
        native_rgb.append(np.asarray(image).copy())
        rgb.append(np.asarray(image.resize((256,144),Image.Resampling.BOX)).copy())
        mask=np.load(mp,allow_pickle=False)
        assert mask.shape==(2,360,640) and np.isin(mask,[-1,0,1]).all()
        native_masks.append(mask)
        cells=torch.as_tensor(mask,device='cuda').reshape(2,18,20,32,20)
        pos=(cells==1).any(dim=4).any(dim=2); unk=(cells==-1).any(dim=4).any(dim=2)
        support.append(torch.where(pos,1,torch.where(unk,-1,0)).to(torch.int8).cpu().numpy())
        near.append(row['body_head_visible_targets'])
        records.append(dict(sample_index=i,sample_id=case['name'],group_id=case['group_id'],
            source_partition=case['source_partition'],source_role=case['source_role'],source_exposure=case['source_exposure'],
            appearance=case['appearance'],relation=case['relation'],family=case['condition']['family'],
            parent_case_name=case['parent_case_name'],original_case_name=case['parent_case_name'],
            camera=case['camera'],native_near=row['body_head_visible_targets'],
            rgb_sha256=sha(ip),mask_sha256=sha(mp),native_rgb_sha256=sha(ip),native_mask_sha256=sha(mp)))
    comparisons=[]
    for i,case in enumerate(spec['cases']):
        if case['appearance']=='A': continue
        ai=next(j for j,c in enumerate(spec['cases']) if c['group_id']==case['group_id'] and c['relation']==case['relation'] and c['appearance']=='A')
        a=spec['cases'][ai]
        geom_equal=all(case[k]==a[k] for k in ('camera','wearer','floor_z_m','geometric_contact','geometric_intrusion'))
        geometry=lambda c:[{k:v for k,v in o.items() if k!='material_asset'} for o in c['objects']]
        geom_equal=geom_equal and geometry(case)==geometry(a)
        different=int(np.count_nonzero(native_rgb[i]!=native_rgb[ai]))
        delta=float(np.abs(native_rgb[i].astype(np.float32)-native_rgb[ai]).mean())
        mask_equal=bool(np.array_equal(native_masks[i],native_masks[ai]))
        comparisons.append(dict(sample_index=i,baseline_index=ai,appearance=case['appearance'],
            geometry_equal=geom_equal,native_support_unknown_exactly_equal=mask_equal,
            rgb_different_channels=different,rgb_mean_absolute_delta=delta,
            box_rgb_different_channels=int(np.count_nonzero(rgb[i]!=rgb[ai]))))
    failures=[]
    for row in comparisons:
        if not row['geometry_equal'] or not row['native_support_unknown_exactly_equal'] or not row['box_rgb_different_channels']:
            failures.append(row['sample_index'])
    admission=dict(status='PASS' if not failures else 'FAIL',failed_indices=failures,comparisons=comparisons,
        native_label_counts=np.asarray(near).sum(axis=0).tolist(),
        labels_match_intended=all(row==[0.,float(c['relation']=='HEAD_ONLY')] for c,row in zip(spec['cases'],near)),
        contract='Native full-resolution BODY/HEAD support including UNKNOWN exact equality across appearance; RGB actual change; all diagnostic labels evaluator-only')
    write(output/'appearance-admission.json',admission)
    arrays={}
    for name,array in [('rgb',np.stack(rgb)),('near',np.asarray(near,np.float32)),('support',np.stack(support)),('native_support',np.stack(native_masks))]:
        rel=f'{"model" if name=="rgb" else "evaluator"}/diagnostic/{name}.npy'
        path=output/rel;path.parent.mkdir(parents=True,exist_ok=True);np.save(path,array,allow_pickle=False)
        arrays[name]=dict(path=rel,sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype),bytes=path.stat().st_size)
    ids=list(range(len(records)))
    write(output/'evaluator/diagnostic.json',dict(sample_indices=ids,group_ids=[r['group_id'] for r in records],near=arrays['near'],support=arrays['support'],native_support=arrays['native_support']))
    write(output/'evaluator/groups.json',records)
    write(output/'source/spec.json',spec)
    manifest=dict(schema='city-training-cache-v1',status=admission['status'],role='DIAGNOSTIC_ONLY',
        partitions=dict(diagnostic=dict(sample_indices=ids,rgb=arrays['rgb'])),source_capture=str(capture),
        source_spec_sha256=sha(capture/'source/spec.json'),native_verification_sha256=sha(capture/'world-verification.json'),
        source_sha256=sha(Path(__file__)),backend='CUDA',device=torch.cuda.get_device_name(),
        native_positive_counts=np.asarray(near).sum(axis=0).tolist(),admission=admission)
    write(output/'manifest.json',manifest)
    print(json.dumps(dict(status=admission['status'],failed_indices=failures,labels_match_intended=admission['labels_match_intended'],frames=len(records))))
    if failures or not admission['labels_match_intended']:
        raise ValueError('Appearance admission failed; preserve all rows and inspect without model access')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--source',type=Path,required=True);g.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('build');b.add_argument('--capture',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='generate': generate(args.source,args.output)
    else: build(args.capture,args.output)
