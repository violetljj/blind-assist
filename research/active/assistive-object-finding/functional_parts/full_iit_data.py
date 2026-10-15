"""Resumable official IIT archive and full native train/val census; test GT stays opaque."""
from __future__ import annotations
import argparse, collections, concurrent.futures, hashlib, io, json, os, time, zipfile, struct, zlib, re
from pathlib import Path
import numpy as np
from PIL import Image
import requests
from acquire_data import Source, OBJECTS
import threading

def write_json(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(tmp,path)

def audit_existing(out, audit=None):
    """Read only prepared train/val; never mutate frozen training records."""
    rows=[json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()]
    shas=collections.defaultdict(list);prefixes=collections.defaultdict(collections.Counter)
    totals={sp:{'all_pairs':collections.Counter(),'samebox_nonoverlap_pairs':collections.Counter(),'overlap_ambiguous_pairs':collections.Counter(),'boxes':0,'overlap_boxes':0} for sp in ('train','val')}
    details=[]
    for r in rows:
        match=re.match(r'^((?:img)?[0-9]+)_[0-9]+$',r['source_id'])
        if match:prefixes[match.group(1)][r['split']]+=1
        if r['split']=='test':continue
        assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['image_sha256']
        assert hashlib.sha256(Path(r['mask']).read_bytes()).hexdigest()==r['mask_sha256']
        shas[r['image_sha256']].append({'source_id':r['source_id'],'split':r['split']})
        with Image.open(r['mask']) as im:arr=np.asarray(im)
        h,w=arr.shape;st=totals[r['split']]
        for idx,o in enumerate(r['objects']):
            x1,y1,x2,y2=o['box_native'];x1,x2=max(0,x1),min(w,x2+1);y1,y2=max(0,y1),min(h,y2+1)
            hist=np.bincount(arr[y1:y2,x1:x2].reshape(-1),minlength=10)
            overlap=any(j!=idx and max(x1,q['box_native'][0])<min(x2,q['box_native'][2]+1) and max(y1,q['box_native'][1])<min(y2,q['box_native'][3]+1) for j,q in enumerate(r['objects']))
            labs=np.flatnonzero(hist[1:])+1;st['boxes']+=1;st['overlap_boxes']+=overlap
            for i in range(len(labs)):
                for j in range(i+1,len(labs)):
                    key=f'{o["category"]}:{labs[i]}+{labs[j]}'
                    st['all_pairs'][key]+=1;st['overlap_ambiguous_pairs' if overlap else 'samebox_nonoverlap_pairs'][key]+=1
            details.append({'source_id':r['source_id'],'split':r['split'],'object_index':idx,**o,'class_pixel_counts':{str(k):int(v) for k,v in enumerate(hist)},'bbox_overlaps_other':overlap})
    with (out/'bbox_census_details.jsonl').open('w',encoding='utf-8') as f:
        for r in details:f.write(json.dumps(r)+'\n')
    write_json(out/'bbox_census_audit.json',{'splits':totals,'minimum_pixels_for_pair':1,'pixel_coordinate_convention':'inclusive bbox xmax/ymax','not_role_truth':True,'no_test_mask_read':True})
    crossing=[v for v in shas.values() if {x['split'] for x in v}=={'train','val'}]
    write_json(out/'train_val_identity_audit.json',{'exact_rgb_sha_cross_split_groups':crossing,'all_prepared_image_mask_hashes_verified':True,'filename_capture_prefix_counts':prefixes,'prefix_train_val_shared':[k for k,v in prefixes.items() if v['train'] and v['val']],'prefix_caveat':'heuristic only; native split not scene-independent','no_test_GT_pixels_read':True})
    if audit is not None:
        source=Source('IIT-AFF',audit,out,time.monotonic()+60,{'bytes':0,'lock':threading.Lock()})
        checks=[]
        with (out/'IIT_Affordances_2017.zip').open('rb') as f:
            for split in ('train','val'):
                rr=[r for r in rows if r['split']==split]
                for index in np.linspace(0,len(rr)-1,4,dtype=int):
                    r=rr[index];entry=source.entries[r['archive_mask_member']];f.seek(entry['offset'])
                    fields=struct.unpack('<4s5H3L2H',f.read(30));n,x=fields[-2:];f.read(n+x)
                    packed=f.read(entry['compressed_size']);raw=zlib.decompress(packed,-15) if fields[3]==8 else packed
                    assert f'{zlib.crc32(raw):08x}'==entry['crc32']
                    independent=np.loadtxt(io.BytesIO(raw),dtype=np.uint8)
                    with Image.open(r['mask']) as im:saved=np.asarray(im)
                    assert np.array_equal(independent,saved)
                    checks.append({'source_id':r['source_id'],'split':split,'exact_equal':True})
        write_json(out/'independent_native_parse_audit.json',{'checks':checks,'method':'np.loadtxt vs original np.fromstring conversion','test_GT_pixels_read':False})
    print(json.dumps({'audit':'PASS','train_val_hash_duplicate_groups':len(crossing),'boxes_audited':len(details)}),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True); p.add_argument('--workers',type=int,default=6)
    p.add_argument('--metadata-only',action='store_true')
    p.add_argument('--audit-existing',action='store_true',help='Verify sealed inputs and emit separate bbox/identity census without changing records')
    a=p.parse_args()
    out=a.root/'data/iit'; out.mkdir(parents=True,exist_ok=True)
    if a.audit_existing:
        audit_existing(out,a.audit);return
    if (out/'records.jsonl').exists() and not a.metadata_only:
        # Training manifests are hash-frozen by the parent; reruns may validate,
        # but must not silently rewrite them with a changed census schema.
        audit_existing(out,a.audit);return
    start=time.monotonic(); cpu=time.process_time(); old=a.audit/'iit_aff_audit'
    sets={k:set(x.strip().removesuffix('.jpg') for x in (old/(k+'.txt')).read_text().splitlines() if x.strip()) for k in ('train_and_val','val','test')}
    assert sets['val']<=sets['train_and_val'] and not sets['test']&sets['train_and_val']
    sets['train']=sets['train_and_val']-sets['val']
    info={'counts':{k:len(v) for k,v in sets.items()},'test_GT_pixels_read':False,'train_excludes_native_val':True}
    write_json(out/'split_census.json',info); print(json.dumps(info),flush=True)
    if a.metadata_only:return
    budget={'bytes':0,'lock':threading.Lock()}
    s=Source('IIT-AFF',a.audit,out,time.monotonic()+7200,budget); s.connect()
    archive=out/'IIT_Affordances_2017.zip'; markers=out/'download_chunks'; markers.mkdir(exist_ok=True)
    # The first 944 MB contains all labels/boxes/RGB. The other 4.5 GB is depth,
    # which this RGB study does not use. Keep test labels compressed and opaque.
    needed_end=max(e['offset']+30+len(n.encode())+4096+e['compressed_size'] for n,e in s.entries.items() if '/rgb/' in n)
    chunk=1024*1024; ranges=[(i,min(needed_end-1,i+chunk-1)) for i in range(0,needed_end,chunk)]
    write_json(out/'download_plan.json',{'archive_full_size':s.size,'needed_prefix_bytes':needed_end,'status':'SPARSE_PARTIAL_ARCHIVE_NOT_A_COMPLETE_ZIP','depth_not_downloaded':True})
    if not archive.exists():
        with archive.open('wb') as f:f.truncate(needed_end)
    assert archive.stat().st_size>=needed_end
    receipts=[]; completed=0
    def fetch(pair):
        lo,hi=pair; marker=markers/f'{lo}.json'
        if marker.exists():
            m=json.loads(marker.read_text())
            with archive.open('rb') as f:f.seek(lo); b=f.read(hi-lo+1)
            if hashlib.sha256(b).hexdigest()==m['sha256']:return {'lo':lo,'resumed':True,'bytes':0}
        url,params=s.form
        for attempt in range(5):
            record={'lo':lo,'hi':hi,'attempt':attempt,'bytes':0}
            try:
                with requests.get(url,params=params,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(15,90)) as r:
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{s.size}':raise ValueError(f'Range response {r.status_code}: {r.headers.get("Content-Range")}')
                    data=bytearray()
                    for b in r.iter_content(1024*1024):
                        data.extend(b);record['bytes']+=len(b)
                        if len(data)>hi-lo+1:raise ValueError('Oversized range')
                if len(data)!=hi-lo+1:raise ValueError('Short range')
                with archive.open('r+b') as f:f.seek(lo);f.write(data);f.flush()
                digest=hashlib.sha256(data).hexdigest()
                write_json(marker,{'lo':lo,'hi':hi,'sha256':digest})
                record['status']='SUCCESS';return record
            except (requests.RequestException,ValueError) as e:
                record['status']=type(e).__name__;record['error']=str(e)
                print(json.dumps({'stage':'download_retry',**record}),flush=True)
                if attempt==4:raise
            finally:
                with budget['lock']:receipts.append(record);budget['bytes']+=record['bytes']
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for r in pool.map(fetch,ranges):
                completed+=1
                if completed%4==0 or completed==len(ranges):
                    prog={'stage':'download','chunks':completed,'total':len(ranges),'received_bytes':budget['bytes'],'wall_s':time.monotonic()-start}
                    write_json(out/'progress.json',prog);print(json.dumps(prog),flush=True)
    finally:write_json(out/'download_receipt.json',{'requests':receipts,'received_bytes':budget['bytes'],'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu})
    with archive.open('rb') as f:
        hsh=hashlib.sha256();remaining=needed_end
        while remaining:
            b=f.read(min(1024*1024,remaining));hsh.update(b);remaining-=len(b)
    write_json(out/'archive_receipt.json',{'full_size':s.size,'available_prefix_bytes':needed_end,'prefix_sha256':hsh.hexdigest(),'all_required_chunks_complete':True,'complete_zip':False})
    images=out/'images';masks=out/'masks';images.mkdir(exist_ok=True);masks.mkdir(exist_ok=True)
    stats={sp:{'images':0,'pixels':collections.Counter(),'class_images':collections.Counter(),'objects':collections.Counter(),'bbox_pairs':collections.Counter(),'bbox_pair_samebox_nonoverlap':collections.Counter(),'bbox_pair_overlap_ambiguous':collections.Counter(),'multi_object_images':0,'overlapping_bbox_images':0,'empty_foreground_images':0,'bbox_no_foreground':0,'unassigned_foreground_pixels':0} for sp in ('train','val')}
    records=[]
    class PrefixMembers:
        def __enter__(self):self.f=archive.open('rb');return self
        def __exit__(self,*args):self.f.close()
        def read(self,name):
            e=s.entries[name];self.f.seek(e['offset']);header=self.f.read(30);fields=struct.unpack('<4s5H3L2H',header)
            assert fields[0]==b'PK\x03\x04';n,x=fields[-2:];assert self.f.read(n).decode()==name;self.f.read(x)
            packed=self.f.read(e['compressed_size']);raw=zlib.decompress(packed,-15) if fields[3]==8 else packed
            assert len(raw)==e['size'] and f'{zlib.crc32(raw):08x}'==e['crc32'];return raw
    with PrefixMembers() as z:
        for sp in ('train','val','test'):
            for index,sid in enumerate(sorted(sets[sp])):
                imp=images/(sid+'.jpg'); mp=masks/(sid+'.png')
                imember=f'IIT_Affordances_2017/rgb/{sid}.jpg';mmember=f'IIT_Affordances_2017/affordances_labels/{sid}.txt'
                obmember=f'IIT_Affordances_2017/object_labels/{sid}.txt'
                boxes=[[int(v) for v in line.split()] for line in z.read(obmember).decode().splitlines() if line.strip()]
                assert all(len(b)==5 for b in boxes)
                objects=[{'category_id':b[0],'category':OBJECTS[b[0]],'box_native':b[1:]} for b in boxes]
                rec={'source_id':sid,'image':str(imp.resolve()),'mask':str(mp.resolve()) if sp!='test' else None,'split':sp,'native_train_and_val':sid in sets['train_and_val'],'objects':objects,'source_group':None,'source_group_status':'not_supplied_by_native_metadata','archive_image_member':imember,'archive_mask_member':mmember,'mask_pixel_read':sp!='test'}
                capture=re.match(r'^((?:img)?[0-9]+)_[0-9]+$',sid)
                if capture:
                    rec['source_group']='filename_capture:'+capture.group(1)
                    rec['source_group_status']='filename_heuristic_not_verified_scene_or_instance'
                if sp=='test':
                    rec['image_materialized']=False;rec['test_storage']='opaque zip member; no GT decompression';records.append(rec);continue
                if not imp.exists():imp.write_bytes(z.read(imember))
                with Image.open(imp) as im:w,h=im.size
                if mp.exists():arr=np.asarray(Image.open(mp))
                else:
                    raw=z.read(mmember);arr=np.fromstring(raw.decode('ascii'),sep=' ',dtype=np.int16).reshape(h,w)
                    if arr.min()<0 or arr.max()>9:raise ValueError('Native class out of bounds')
                    arr=arr.astype('uint8');Image.fromarray(arr).save(mp)
                assert arr.shape==(h,w) and arr.dtype==np.uint8
                st=stats[sp];st['images']+=1
                hist=np.bincount(arr.reshape(-1),minlength=10);st['pixels'].update({i:int(c) for i,c in enumerate(hist)});st['class_images'].update(np.flatnonzero(hist).tolist())
                st['objects'].update(o['category'] for o in objects);st['multi_object_images']+=len(objects)>1;st['empty_foreground_images']+=not np.any(arr)
                coverage=np.zeros((h,w),dtype=np.uint16)
                for object_index,o in enumerate(objects):
                    x1,y1,x2,y2=o['box_native'];x1,x2=max(0,x1),min(w,x2+1);y1,y2=max(0,y1),min(h,y2+1)
                    region=arr[y1:y2,x1:x2];labs=np.unique(region);labs=labs[labs>0].tolist()
                    o['bbox_native_affordance_ids']=labs;o['association']='semantic mask inside native bbox; not instance-exclusive GT'
                    o['class_pixel_counts']={int(k):int(v) for k,v in enumerate(np.bincount(region.reshape(-1),minlength=10))}
                    overlap=False
                    for other_index,other in enumerate(objects):
                        if other_index==object_index:continue
                        a1,b1,a2,b2=other['box_native']
                        overlap |= max(x1,a1)<min(x2,a2+1) and max(y1,b1)<min(y2,b2+1)
                    o['bbox_overlaps_other']=bool(overlap)
                    st['bbox_no_foreground']+=len(labs)==0
                    for i in range(len(labs)):
                        for j in range(i+1,len(labs)):
                            key=f'{o["category"]}:{labs[i]}+{labs[j]}'
                            st['bbox_pairs'][key]+=1
                            st['bbox_pair_overlap_ambiguous' if overlap else 'bbox_pair_samebox_nonoverlap'][key]+=1
                    coverage[y1:y2,x1:x2]+=1
                st['overlapping_bbox_images']+=bool(np.any(coverage>1));st['unassigned_foreground_pixels']+=int(np.count_nonzero((arr>0)&(coverage==0)))
                rec.update(width=w,height=h,image_materialized=True,image_sha256=hashlib.sha256(imp.read_bytes()).hexdigest(),mask_sha256=hashlib.sha256(mp.read_bytes()).hexdigest())
                records.append(rec)
                if (index+1)%200==0:
                    print(json.dumps({'stage':'prepare','split':sp,'count':index+1,'total':len(sets[sp])}),flush=True)
            if sp!='test':
                with (out/f'{sp}_records.jsonl').open('w',encoding='utf-8') as f:
                    for r in records:
                        if r['split']==sp:f.write(json.dumps(r)+'\n')
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    sha_groups=collections.defaultdict(list)
    prefixes=collections.defaultdict(lambda:collections.Counter())
    for r in records:
        if 'image_sha256' in r:sha_groups[r['image_sha256']].append({'source_id':r['source_id'],'split':r['split']})
        # Numeric/image capture-prefix only: ImageNet synsets are classes, not scenes.
        match=re.match(r'^((?:img)?[0-9]+)_[0-9]+$',r['source_id'])
        if match:prefixes[match.group(1)][r['split']]+=1
    crossing=[v for v in sha_groups.values() if {x['split'] for x in v}=={'train','val'}]
    write_json(out/'train_val_identity_audit.json',{'exact_rgb_sha_cross_split_groups':crossing,'filename_capture_prefix_counts':prefixes,'prefix_train_val_shared':[k for k,v in prefixes.items() if v['train'] and v['val']],'prefix_caveat':'heuristic capture naming; no physical instance/scene identity is supplied; native split is not scene-independent'})
    write_json(out/'native_census.json',{'splits':stats,'counts':info['counts'],'test_GT_pixels_read':False,'instance_pair_claim':'bbox-co-occurrence at >=1 pixel per class only; even nonoverlap is not native role-pair GT; overlapping boxes can mix instances','box_convention':'xmin/ymin/xmax/ymax interpreted as inclusive endpoints, matching prior project loader','wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu})
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
