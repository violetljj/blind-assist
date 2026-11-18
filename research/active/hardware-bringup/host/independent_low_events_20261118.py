"""Lock reference-only object judgments and score fixed arms on those same objects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/independent-low-20261118/scoring'
PROTOCOL = {
    'scope': 'Outdoors, real images, previously unused recording for the frozen decision. Reference-only screening; independent recording does not imply independent site.',
    'reference': 'Walk-aligned visible corridor |x| <= 0.35 m. Main freestanding low obstacle / curb / stair face 0.15 <= height < 0.50 m; secondary recognizable face 0.05 <= height < 0.15 m. Near observation forward 0.30..1.50 m, earlier approach evidence may be as far as 2.50 m. No absolute distance assigned from RGB alone.',
    'entities': 'Outline the visible object surface in its native RGB; intersect with the same physical foreground reference projection at near range. Positive requires >=0.005 m2 opened reference support on the main face. Disclose sparse LiDAR boundaries or missing optical calibration. An edge on a traversable coplanar surface, a cast shadow, an unresolved surface, a person or the base of a tall object does not qualify as an isolated low obstacle. Link steps of the same stair flight, adjacent samples, rerecordings of the same installation and repeated views by layout or mapped position. Track site, object and encounter IDs separately.',
    'encounters': 'An approaching encounter requires at least two timed views of the same object in the path with calibrated forward distance falling by >0.15 m and a visible qualifying near point. Single observations, nominal camera movement without range corroboration, side passes and departures are separate. Ambiguous revisits stay linked, with ambiguity reported.',
    'matching': 'Frozen bands and whole-scene alarms unchanged. A matched-band hit requires that band to alarm and its after-opening predicted support inside the same object reference footprint and RGB polygon to reach 0.005 m2 in at least one eligible near observation. LOW primary, any independently alarming band with sufficient matched support reported separately. Evaluate objects and encounters using the same near frames across arms.',
    'negatives': 'Same time and observed corridor for each arm; separately report strict LOW zero and strict all-band zero before opening and missing/ambiguous coverage. True empty requires independently visible foreground and available depth, not absence of samples. Compare paired negative sample rate and discordant counts. Event rates require a contiguous native-clock negative interval (indices and timestamps); sampled isolated images are reported per sample only. Different empirical rates, saturated negatives, and absence of a powered margin cannot establish equivalence at equal false-alarm burden.',
    'aggregation': 'Independent results tabulated per source; each distinct object once, each documented encounter once. Repeated frames, steps, slices and replays do not increase physical object count. Historic development examples and sparse/illustrative references not promoted. No sensitivity claim without denominator and approach checks.',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(filename, document):
    payload = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    OUT.mkdir(parents=True,exist_ok=True)
    path = OUT / filename
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Different contents already locked at {path}')
    else:
        with path.open('xb') as dest:
            dest.write(payload)
    return digest(path)


def lock(path):
    doc = json.loads(path.read_text(encoding='utf-8-sig'))
    seen = set()
    objects = set()
    for case in doc['cases']:
        key = case['id']
        assert key not in seen, key
        seen.add(key)
        assert case['decision'] in ('eligible','secondary','defer','exclude'), key
        assert case['source'] and case['reason']
        if case['decision'] in ('eligible','secondary'):
            assert case['object_id'] and case['encounter_id']
            objects.add(case['object_id'])
            assert case['reference_area_m2'] >= .005
            assert .30 <= case['near_forward_m'] <= 1.50
            assert case['polygon'] and case['reference_footprint']
            if case['approach']:
                assert case['early_t_s'] < case['near_t_s']
                assert case['near_forward_m']+.15 < case['early_forward_m'] <= 2.50
    sources = {}
    for p in doc['source_files']:
        file = Path(p)
        sources[str(file)] = digest(file)
    protocol_hash = write_once('protocol.json',PROTOCOL)
    data = dict(protocol_sha256=protocol_hash,source_sha256=sources,
                screened_case_count=len(seen),unique_included_objects=len(objects),**doc)
    return write_once(path.stem+'-locked.json',data)


def contact(path, picked=False):
    """Blind contact sheet of every sufficiently near reference candidate, including weak geometry."""
    import cv2
    import numpy as np
    tiles=[]
    rows=json.loads(path.read_text(encoding='utf-8-sig'))
    selected={}
    if picked:
        picks_path=path.with_name(path.name.replace('screen_','picked_',1))
        selected=json.loads(picks_path.read_text(encoding='utf-8-sig'))
    for row in rows:
        if picked and (row['seq'] not in selected or selected[row['seq']]['frame']!=row['frame']):
            continue
        near=([selected[row['seq']]['component']] if picked else
              [c for c in row.get('components',[]) if c['forward_min']<=1.50])
        if not near:
            continue
        if not Path(row['rgb']).is_file():
            continue
        # SANPO's full optical image is 2208x1242; stereo candidate coordinates are 1280x720.
        rgb=cv2.resize(cv2.imread(row['rgb']), (1280,720), interpolation=cv2.INTER_AREA)
        source=cv2.resize(rgb,(384,216),interpolation=cv2.INTER_AREA)
        for c in near:
            poly=np.array(c['polygon'],np.int32)
            cv2.polylines(rgb,[poly],True,(5,60,250),4,cv2.LINE_AA)
        view=cv2.resize(rgb,(384,216),interpolation=cv2.INTER_AREA)
        panel=cv2.hconcat([source,view])
        panel=cv2.copyMakeBorder(panel,26,0,0,0,cv2.BORDER_CONSTANT,value=(29,29,29))
        text=f"{row['seq'][:7]} / {row['frame']}  " + ', '.join(f"{c['band']} zmin {c['forward_min']:.2f}" for c in near)
        cv2.putText(panel,text,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        tiles.append(panel)
    for i in range(0,len(tiles),6):
        image=cv2.vconcat(tiles[i:i+6])
        assert cv2.imwrite(str(OUT/f'{path.stem}-{"picked" if picked else "blind"}-{i//6}.jpg'),image)
    print(len(tiles),'reference candidate frames rendered')


def reference_review(manifests, prefixes, audit_all=False):
    """Render all native reference votes, prior to object-polygon adjudication."""
    import cv2
    import numpy as np
    import independent_low_replay_20261118 as replay
    from sanpo_ground_alert_replay_20261118 import evaluator
    records={}
    all_geometry=[]
    for manifest in manifests:
        dataset,rows=replay.inputs(manifest)
        directory=replay.OUT/replay.sha(manifest)[:16]
        if not (directory/'reference.json').exists():
            continue
        ref={(r['seq'],r['i']):r for r in replay.read(directory/'reference.json')['rows']}
        for row in rows:
            seq,i=row['seq'],row['i']
            if (not audit_all and not seq.startswith(tuple(prefixes))) or (seq,i) in records:
                continue
            rr=ref[seq,i]
            if not rr['ground']:
                continue
            f=replay.load_frame(dataset,row,directory)
            h,w=f.ref_depth.shape
            reg=evaluator(rr['ground'])
            z=f.ref_depth.ravel()
            valid=np.isfinite(z)&(z>.3)&(z<10)
            z=np.where(valid,z,0)
            xyz=replay.core.rays(f.K,h,w)*z[:,None]
            above,along,planar=replay.core._corridor(xyz,valid,reg['axes'],reg['ground'][1])
            areas=replay.core.pixel_area(f.K,h,w,z)
            measure={}
            masks={}
            for band,(lo,hi,col) in {'LOW':(.15,.50,(0,210,245)),'shallow':(.05,.15,(255,170,0))}.items():
                support=planar&(above>=lo)&(above<hi)
                for near,limit in [('near',1.5),('full',2.5)]:
                    mask=replay.core._open(support&(along<=limit),h,w)
                    measure[band+'_'+near]=[round(float(areas[mask].sum()),6),
                                            round(float(np.percentile(along[mask],10)),4) if mask.any() else None]
                    if near=='near':
                        masks[band]=mask.reshape(h,w)
            if audit_all:
                all_geometry.append(dict(seq=seq,i=i,**measure))
                if max(measure['LOW_near'][0],measure['shallow_near'][0])<.005:
                    continue
            base=replay.image(row,target=(384,216))
            panel=base.copy()
            for band,col in [('LOW',(0,210,245)),('shallow',(255,170,0))]:
                m=cv2.resize(masks[band].astype('uint8'),(384,216),interpolation=cv2.INTER_AREA)
                contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel,contours,-1,col,1,cv2.LINE_AA)
            maskfile=OUT/f'ref-{seq[:10]}-{i}.npz'
            if not audit_all:
                np.savez_compressed(maskfile,shape=np.array([h,w]),**{k:np.packbits(v,axis=-1,bitorder='little') for k,v in masks.items()})
            draw=cv2.copyMakeBorder(cv2.hconcat([base,panel]),24,0,0,0,cv2.BORDER_CONSTANT)
            label=f"{seq[:8]} #{i}  near LOW {measure['LOW_near'][0]:.4f} SH {measure['shallow_near'][0]:.4f}"
            cv2.putText(draw,label,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            records[seq,i]=(draw,dict(seq=seq,i=i,t=row['t'],shape=[h,w],manifest=str(manifest.resolve()),rgb=str(row['rgb']),
                                      maskfile=str(maskfile),geometry=measure))
    if audit_all:
        shortlist=sorted(records)
        for j in range(0,len(shortlist),12):
            a=[cv2.resize(records[k][0],(480,150)) for k in shortlist[j:j+12]]
            if len(a)%2:
                a.append(np.zeros_like(a[-1]))
            cv2.imwrite(str(OUT/f'formal-all-near-{j//12}.jpg'),cv2.vconcat([cv2.hconcat(a[k:k+2]) for k in range(0,len(a),2)]))
        (OUT/'formal-all-near.json').write_text(json.dumps(dict(all_reference=all_geometry,candidates=[records[k][1] for k in shortlist]),indent=1))
        print(len(all_geometry),'evaluable reference frames,',len(shortlist),'qualifying near-region candidates')
        return
    for seq in sorted({s for s,i in records}):
        group=sorted(i for s,i in records if s==seq)
        for j in range(0,len(group),5):
            tiles=[records[seq,i][0] for i in group[j:j+5]]
            assert cv2.imwrite(str(OUT/f'ref-{seq[:10]}-sheet-{j//5}.jpg'),cv2.vconcat(tiles))
    data=[v[1] for v in records.values()]
    for seq in sorted({s for s,i in records}):
        (OUT/f'ref-{seq[:10]}-index.json').write_text(json.dumps([r for r in data if r['seq']==seq],indent=1))
    print(len(data),'label-only reference rows')


def oxford():
    """Reviewed outlines in 480x640 rectified RGB, constrained to visible TLS riser faces."""
    import cv2
    import numpy as np
    import independent_low_replay_20261118 as replay
    from sanpo_ground_alert_replay_20261118 import evaluator
    local=OUT.parent/'local'
    manifest=local/'filled_generic_ground.json'
    ds,inputs=replay.inputs(manifest)
    directory=replay.OUT/replay.sha(manifest)[:16]
    ref={(x['seq'],x['i']):x for x in replay.read(directory/'reference.json')['rows']}
    rows={r['i']:r for r in inputs}
    # For a stair flight the enclosed geometry is further restricted to upright risers.
    polys={
        15:[[96,452],[324,360],[326,407],[108,499]],
        18:[[102,499],[373,389],[376,447],[111,559]],
        19:[[108,512],[375,395],[379,454],[104,569]],
        20:[[224,539],[478,413],[478,450],[301,539]],
        108:[[90,377],[460,423],[479,556],[106,454]],
        113:[[65,406],[477,453],[479,572],[57,494]],
        114:[[55,417],[477,463],[479,581],[42,511]],
        115:[[52,431],[477,480],[479,588],[36,522]],
    }
    cases=[]
    source_files=[manifest,local/'filled_rows.json']
    for candidates,early,entity,kind in [([18,19,20],15,'blenheim-courtyard-front-edge','single raised curb / terrace edge'),
                                         ([113,114,115],108,'blenheim-building-stair-flight','extension: multi-step staircase')]:
        evidence=[]
        for i in [early]+candidates:
            row=rows[i]
            f=replay.load_frame(ds,row,directory)
            reg=evaluator(ref[row['seq'],i]['ground'])
            h,w=f.ref_depth.shape
            z=f.ref_depth.ravel()
            ok=np.isfinite(z)&(z>.3)&(z<10)
            z=np.where(ok,z,0)
            rays=replay.core.rays(f.K,h,w)
            height,along,scope=replay.core._corridor(rays*z[:,None],ok,reg['axes'],reg['ground'][1])
            areas=replay.core.pixel_area(f.K,h,w,z)
            polygon=np.array(polys[i],np.int32)
            poly=np.zeros((h,w),np.uint8)
            cv2.fillPoly(poly,[polygon],1)
            facefile=Path(str(row['rgb']).replace('_rgb.jpg','_face.png'))
            face=cv2.imread(str(facefile),cv2.IMREAD_UNCHANGED)
            assert face is not None
            valid=scope & (height>=.15)&(height<.50)&poly.ravel().astype(bool)&((face.ravel()&1)==1)
            metric = {}
            for k,zmax in [('near',1.5),('full',2.5)]:
                support=replay.core._open(valid&(along<=zmax),h,w)
                metric[k]={'area_m2':float(areas[support].sum()),'forward_m':float(np.percentile(along[support],10)) if support.any() else None}
                if k=='near':
                    nearmask=support.reshape(h,w)
            target=OUT/f'oxford-{i}-face-near.png'
            assert cv2.imwrite(str(target),nearmask.astype(np.uint8)*255)
            source_files.extend([row['rgb'],facefile,target])
            evidence.append(dict(i=i,seq=row['seq'],t_s=row['t'],polygon=polys[i],reference_footprint=str(target),
                                 reference_mask_sha256=digest(target),rgb=str(row['rgb']),**metric))
        near=[x for x in evidence[1:] if x['near']['area_m2']>=.005]
        assert near
        best=min(near,key=lambda x:x['near']['forward_m'])
        assert evidence[0]['full']['area_m2']>=.005
        cases.append(dict(id=entity,source='Oxford Spires',decision='eligible',object_id=entity,
                     encounter_id=entity+'-first',category=kind,reason='Visible distinct TLS riser in the camera corridor.',
                     near_t_s=best['t_s'],early_t_s=evidence[0]['t_s'],near_forward_m=best['near']['forward_m'],
                     early_forward_m=evidence[0]['full']['forward_m'],approach=True,polygon=best['polygon'],
                     reference_footprint=best['reference_footprint'],reference_area_m2=best['near']['area_m2'],
                     observations=evidence,matched_indices=[x['i'] for x in near],manifest=str(manifest)))
    data=dict(source_files=sorted({str(p) for p in source_files}),cases=cases)
    write_once('oxford-review.json',data)
    print(lock(OUT/'oxford-review.json'))


def lamar(extra=False):
    """Two visible LaMAR structures, polygon drawn on the native sensor geometry."""
    import cv2
    import numpy as np
    import independent_low_replay_20261118 as replay
    local=OUT.parent/'alt-native/case_windows'
    polygons={
      'lin-stone-bollards':('extension: row of short stone bollards',local/'3187620675/candidates.json',[
        (3186920991,[(86,143),(90,105),(96,97),(105,96),(114,101),(118,145),(114,153),(94,156),(86,151)]),
        (3187620675,[(43,191),(45,146),(48,138),(58,129),(71,127),(82,132),(89,140),(93,191)]),
        (3187720630,[(28,191),(30,155),(35,146),(47,140),(60,138),(71,141),(80,149),(85,191)])]),
      'lin-stone-stair-flight':('extension: multi-step staircase',local/'129546927471/landing/candidates.json',[
        (129544728465,[(30,110),(190,95),(191,225),(0,230),(0,162)]),
        (129545228239,[(29,109),(190,98),(191,254),(0,250),(0,172)]),
        (129545428149,[(20,111),(190,104),(191,255),(0,255),(0,172)]),
        (129545728013,[(18,109),(190,105),(191,255),(0,255),(0,155)]),
        (129545927923,[(2,115),(191,110),(191,255),(0,255)])])}
    if extra:
        stair_ids=[15554612632,15554812540,15555312312,15555612175,15555812083,15556111946]
        plaza_ids=[3267801155,3268001064,3268300929]
        # The centre and lower portion of the actual stone treads are visible in every reviewed image.
        stair_poly=[(18,166),(172,168),(190,251),(0,255)]
        plaza_poly=[(6,118),(246,127),(254,191),(0,191)]
        polygons={
          'lin-scaffold-stair-flight':('extension: multi-step staircase',local/'15553513134/landing/candidates.json',
                                       [(i,stair_poly) for i in stair_ids]),
          'lin-plaza-stair-flight':('extension: multi-step staircase',local/'3265202328/landing/candidates.json',
                                    [(i,plaza_poly) for i in plaza_ids])}
    cases=[]
    files=[]
    for entity,(kind,manifest,steps) in polygons.items():
        dataset,inputs=replay.inputs(manifest)
        rows={r['i']:r for r in inputs}
        obs=[]
        files.append(manifest)
        for i,poly in steps:
            row=rows[i]
            results=replay.blind_roi(manifest,row['seq'],i,poly,polygon_size=row['size'],return_masks=True)
            nums={}
            for label in ['near','full']:
                m=results['LOW'][label]
                nums[label]=dict(area_m2=m['area_m2'],forward_m=m['forward_p10_m'])
            footprint=OUT/f'lamar-{i}-face-near.png'
            assert cv2.imwrite(str(footprint),results['LOW']['near']['mask'].astype(np.uint8)*255)
            files.extend([row['rgb'],row['depth'],row['confidence'],footprint])
            obs.append(dict(i=i,seq=row['seq'],t_s=row['t'],polygon=poly,reference_footprint=str(footprint),
                            reference_mask_sha256=digest(footprint),rgb=str(row['rgb']),**nums))
        near=[r for r in obs[1:] if r['near']['area_m2']>=.005]
        assert near and obs[0]['full']['area_m2']>=.005, (entity,obs)
        best=min(near,key=lambda r:r['near']['forward_m'])
        cases.append(dict(id=entity,source='LaMAR LIN',decision='eligible',object_id=entity,encounter_id=entity+'-first',
                          category=kind,reason='Native aligned RGB / depth: independent visible stone facility. Native medium/high depth confidence.',
                          near_t_s=best['t_s'],early_t_s=obs[0]['t_s'],near_forward_m=best['near']['forward_m'],
                          early_forward_m=obs[0]['full']['forward_m'],approach=True,polygon=best['polygon'],
                          reference_footprint=best['reference_footprint'],reference_area_m2=best['near']['area_m2'],
                          observations=obs,matched_indices=[x['i'] for x in near],manifest=str(manifest)))
    name='lamar-additional-stairs-review.json' if extra else 'lamar-review.json'
    write_once(name,dict(source_files=sorted({str(p) for p in files}),cases=cases))
    print(lock(OUT/name))


def sanpo():
    """Seal every near-area-positive frame from both independent recording manifests."""
    audit_file=OUT/'formal-all-near.json'
    audit=json.loads(audit_file.read_bytes())
    p=OUT.parent/'sanpo-new'
    # Reviewed with no alarm overlays: source RGB and formal stereo support only.
    reasons={
        'A-d9':'Lower coat and leg of a walking person.',
        'BIhF':'Metal gate / person overlap in the market, no independent short object.',
        'G5cL':'Walking person occludes the stairs. Near support is on the backpack, coat and trousers; no near area on the staircase.',
        'N2tK':'Seating installation with a tall back; repeated slats and edge of the same bench.',
        'PbY6':'Leg of the foreground person wearing jeans.',
        'Q373':'People or the bottom of a tall gate in a crowd.',
        'ViIT':'Legs / held shopping bag of nearby walking people.',
        'Xlm-':'Leafy slope / roots with motion blur. No distinct visible curb, platform or wheel stop, no same shallow rigid object established.',
        'Y_lA':'Lower portion of a person beside a market stall.',
        'ZEo3':'Narrow soil patch on an uneven trail. No separable face/outline tied to one shallow entity.',
        'nLaq':'Interior train door / rail in motion blur, outside outdoor walking scope.',
        'oTc3':'Round stone platform. Concentric joints, radial cross-joints, curved face and adjacent plaza match the earlier development facility. Conservatively deduplicated.',
        'qb0z':'Lower part of a tall planter / planting in front of the building.',
    }
    cases=[]
    for r in audit['candidates']:
        reason=reasons[r['seq'][:4]]
        cases.append(dict(id=f"{r['seq']}/{r['i']}",source='SANPO real',decision='exclude',reason=reason,geometry=r['geometry'],rgb=r['rgb']))
    images=sorted(OUT.glob('formal-all-near-*.jpg'))
    sources=[audit_file,p/'release.json',p/'release_supplement.json',OUT/'sanpo-prior-1111-new-120-RGB.jpg',*images]
    for kind in ('sparse','dense','approach'):
        sources.append(p/'blind_review'/f'{kind}.json')
        sources.extend(sorted((p/'blind_review').glob(f'{kind}_*.jpg')))
    sources.extend(Path(r['rgb']) for r in audit['candidates'])
    doc=dict(source_files=sorted({str(s) for s in sources}),cases=cases,
             whole_reference_frames=len(audit['all_reference']),candidate_frames_with_sufficient_near_low_or_shallow=len(cases),
             nominated_visual_review='All 73 sparse, 92 stairs/junction and 72 supplementary anchor images were reviewed. The 237 review rows include repeated sessions and objects. Full refined screen includes automatically evaluated LOW and 5-15 cm near support for every reference-evaluable image. Manual re-review on all formally qualifying frames.',
             independent_sessions=49,paired_sampled_frames=703,
             decision='Zero countable new main-class physical entities or approaches on these sparse/targeted recordings. This is a screening failure, not zero sensitivity.')
    write_once('sanpo-review.json',doc)
    print(lock(OUT/'sanpo-review.json'))


def score_locked():
    import cv2
    import numpy as np
    import independent_low_replay_20261118 as replay
    # The independent candidate screens must be sealed before any arm is read.
    assert (OUT/'sanpo-review-locked.json').exists()
    outputs={}
    def prediction(manifest):
        key=digest(manifest)[:16]
        folder=replay.OUT/key
        if key not in outputs:
            p=folder/'predictions.json'
            raw=replay.read(p)
            data=raw['rows']
            outputs[key]=(folder,{(r['seq'],r['i']):r for r in data},digest(p))
        return outputs[key]

    objects=[]
    for name in ('oxford-review','lamar-review','lamar-additional-stairs-review','sanpo-review'):
        sealed=OUT/(name+'-locked.json')
        doc=replay.read(sealed)
        for case in doc['cases']:
            if case['decision'] not in ('eligible','secondary'):
                continue
            manifest=Path(case['manifest'])
            if case['source']=='LaMAR LIN':
                manifest=OUT.parent/'alt-native/frozen/manifest.json'
            folder,rows,_=prediction(manifest)
            scores=[]
            for obs in case['observations']:
                if obs['i'] not in case['matched_indices']:
                    continue
                row=rows[obs['seq'],obs['i']]
                m=Path(obs['reference_footprint'])
                assert digest(m)==obs['reference_mask_sha256']
                roi=cv2.imread(str(m),cv2.IMREAD_GRAYSCALE)
                support=folder/row['support_path']
                with np.load(support,allow_pickle=False) as archive:
                    h,w=map(int,archive['shape'])
                roi=cv2.resize(roi,(w,h),interpolation=cv2.INTER_NEAREST_EXACT)>0
                overlap=replay.roi_areas(support,roi)
                hits={}
                for arm in replay.ARMS:
                    per_band={b:bool(row['evidence'][arm][b]) and overlap[arm][b]['area_m2']>=.005 for b in replay.BANDS}
                    hits[arm]=dict(LOW=per_band['LOW'],ANY=any(per_band.values()),per_band=per_band)
                scores.append(dict(i=obs['i'],t_s=obs['t_s'],hits=hits,matched_support=overlap,whole_scene=row['evidence']))
            assert scores
            objects.append(dict(source=case['source'],category=case['category'],object_id=case['object_id'],
                                encounter_id=case['encounter_id'],approach=case['approach'],label_sha256=digest(sealed),rows=scores,
                                hit={a:{b:any(r['hits'][a][b] for r in scores) for b in ('LOW','ANY')} for a in replay.ARMS}))
    # The same five RGB-suitable LaMAR windows for all arms. Unknown intervals terminate an episode.
    dataset=OUT.parent/'alt-native/frozen'
    lamar_folder,rows,_=prediction(dataset/'manifest.json')
    selected=replay.read(dataset/'controls.json')
    controls={}
    for key in ['near','full','all_evaluable']:
        eligible={tuple(q) for q in selected['eligible'][key]}
        slab={}
        for arm in replay.ARMS:
            for band in ['LOW','ANY']:
                cases=0; doubles=0; segments=0; represented_seconds=0.; prev=None; prev_active=False
                samples=[]
                for r in sorted(selected['ledger'],key=lambda r:(r['seq'],r['ts'])):
                    t=r['ts']*1e-6
                    identity=(r['seq'],r['ts'])
                    if identity not in eligible:
                        prev=None;prev_active=False
                        continue
                    row=rows[identity]
                    active=any(row['evidence'][arm].values()) if band=='ANY' else bool(row['evidence'][arm][band])
                    contiguous=prev is not None and prev[0]==r['seq'] and 0<t-prev[1]<=.5
                    if contiguous:
                        represented_seconds+=t-prev[1]
                    else:
                        segments+=1
                    if active:
                        cases+=1
                        if not (contiguous and prev_active):
                            doubles+=1
                    samples.append(dict(seq=r['seq'],timestamp_us=r['ts'],alert=active))
                    prev,prev_active=(r['seq'],t),active
                slab[arm+'_'+band]=dict(eligible=len(samples),alert_samples=cases,observed_alarm_runs=doubles,
                                        contiguous_sample_segments=segments,observed_first_last_seconds=represented_seconds,samples=samples)
        controls[key]=slab
    data=dict(objects=objects,lamar_empty=controls,paired_sources={k:v[2] for k,v in outputs.items()})
    print(write_once('matched-results.json',data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',type=Path,help='Reference-only judgment JSON to seal, plus hashes of source files')
    parser.add_argument('--contact',type=Path,help='Render SANPO reference-only candidate rows')
    parser.add_argument('--picked',action='store_true',help='Show one geometrically selected anchor per recording incl. distant context')
    parser.add_argument('--reference',type=Path,nargs='+',help='Fixed source manifests with cached reference.json')
    parser.add_argument('--seq',nargs='+',help='Sequence prefixes for reference-only entity review')
    parser.add_argument('--all-candidates',action='store_true',help='Scan both near height bins in every evaluable frame')
    parser.add_argument('--oxford',action='store_true',help='Seal reviewed Oxford objects')
    parser.add_argument('--lamar',action='store_true',help='Seal reviewed LaMAR objects')
    parser.add_argument('--additional-stairs',action='store_true',help='Seal the two additional geographically distinct LaMAR flights')
    parser.add_argument('--score',action='store_true',help='Join sealed target outlines and negative time intervals against the fixed arms')
    parser.add_argument('--sanpo-lock',action='store_true',help='Seal the blinded SANPO near-area-positive screening')
    args = parser.parse_args()
    if args.sanpo_lock:
        sanpo()
    elif args.score:
        score_locked()
    elif args.additional_stairs:
        lamar(True)
    elif args.lamar:
        lamar()
    elif args.oxford:
        oxford()
    elif args.reference:
        reference_review(args.reference,args.seq or [],args.all_candidates)
    elif args.contact:
        contact(args.contact,args.picked)
    else:
        print(lock(args.lock) if args.lock else write_once('protocol.json', PROTOCOL))


if __name__=='__main__':
    main()
