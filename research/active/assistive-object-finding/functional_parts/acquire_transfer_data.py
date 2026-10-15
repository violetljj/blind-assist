"""Metadata-selected IIT hold transfer; never parse new native mask pixels."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from PIL import Image
from acquire_data import OBJECTS, dump, jsonl, sha, iit_index
from acquire_rank_data import phash
from acquire_visual_data import CountedSource

SEED = '20261015-fp-cross-source'

def rank(x):
    return hashlib.sha256((SEED + str(x)).encode()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    a = p.parse_args()
    assert 'artifacts.local' in a.root.as_posix()
    out = a.root.resolve() / 'data'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, out / 'acquire_transfer_data_at_run.py')
    start, cpu = time.monotonic(), time.process_time()
    budget = {'bytes': 0, 'reserved': 0, 'lock': threading.Lock()}
    source = CountedSource('IIT-AFF', a.audit.resolve(), out, start+900, budget)
    status = 'FAILED'
    try:
        boxes = iit_index(source)
        old_paths = sorted((a.pilot / 'data/IIT-AFF/images').glob('*.jpg'))
        audit_rows = json.loads((source.old / 'summary.json').read_text())['samples']
        refpaths = [(p.stem, p) for p in old_paths]
        for r in audit_rows:
            found = list(source.old.glob('*_'+r['native_test_id']+'_image.jpg'))
            assert len(found) == 1
            refpaths.append((r['native_test_id'], found[0]))
        banned = {i for i,p in refpaths}
        assert len(banned) == 43
        work=a.pilot.parent
        extra = sorted((a.pilot/'data/AGD20K/images').glob('*.jpg'))
        extra += sorted((work/'functional-parts-learned-ranking-20261015/data/images').glob('*.jpg'))
        extra += sorted((work/'functional-parts-visual-ranking-20261015/data/images').glob('*.jpg'))
        extra += sorted((a.audit/'agd20k_audit/samples').glob('*_image.jpg'))
        refpaths += [('agd:'+str(p.resolve()),p) for p in extra]
        refs = [(i, sha(p.read_bytes()), (phash(p), phash(p, True))) for i,p in refpaths]
        jsonl(out/'excluded_rgb_fingerprints.jsonl', [{'id':i,'image_path':str(p.resolve()),'sha256':s,'phash64':[f'{h:016x}' for h in hs]} for (i,p),(_,s,hs) in zip(refpaths,refs)])
        remaining = {i:b for i,b in boxes.items() if i not in banned}
        classes = sorted(OBJECTS, key=lambda k:(sum(any(b[0]==k for b in bs) for bs in remaining.values()),k))
        order = {k: sorted([i for i,bs in remaining.items() if any(b[0]==k for b in bs)],key=rank) for k in classes}
        dump(out/'selection_plan.json', {'seed':SEED,'native_test':len(boxes),'old_excluded_ids':sorted(banned),'cross_source_AGD_refs':len(extra),'available':len(remaining),'class_order':classes,'per_class_target':15,'class_candidate_ids':order,'GT_pixels_read':False,'phash_hamming_exclude_lte':6})
        images, labels = out/'images', out/'native_labels'
        images.mkdir(exist_ok=True); labels.mkdir(exist_ok=True)
        network_start = time.monotonic()
        source.deadline = network_start + 900
        source.connect()
        def rgb(identifier):
            member=f'IIT_Affordances_2017/rgb/{identifier}.jpg'
            path=images/(identifier+'.jpg')
            if not path.exists(): path.write_bytes(source.member(member))
            with Image.open(path) as im:
                im.load(); w,h=im.size
            return path,sha(path.read_bytes()),(phash(path),phash(path,True)),w,h,member
        selected, inspected, rejected, used = [],[],[],set(banned)
        for k in classes:
            candidates=[i for i in order[k] if i not in used]
            accepted=0
            for offset in range(0,len(candidates),4):
                batch=candidates[offset:offset+4]
                with ThreadPoolExecutor(max_workers=4) as pool: values=list(pool.map(rgb,batch))
                for identifier,(path,digest,hs,w,h,member) in zip(batch,values):
                    used.add(identifier)
                    distances=[(min((x^y).bit_count() for x in hs for y in ph),i,digest==s) for i,s,ph in refs]
                    near=min(distances)
                    row={'source_image_id':identifier,'object_id':k,'object':OBJECTS[k],'image_path':str(path),'image_sha256':digest,'width':w,'height':h,'source_image_member':member,'phash64_original_flip':[f'{x:016x}' for x in hs],'nearest_reference_id':near[1],'nearest_hamming':near[0]}
                    inspected.append(row)
                    if near[0]<=6 or any(x[2] for x in distances): rejected.append(row); continue
                    selected.append(row); refs.append((identifier,digest,hs)); accepted+=1
                    if accepted==15: break
                if accepted==15: break
            print(json.dumps({'stage':'RGB','class':OBJECTS[k],'class_accepted':accepted,'total':len(selected),'rejected':len(rejected)}),flush=True)
        assert selected
        jsonl(out/'rgb_inspected.jsonl',inspected)
        jsonl(out/'duplicate_exclusions.jsonl',rejected)
        dump(out/'rgb_selection_seal.json',{'selected_ids':[r['source_image_id'] for r in selected],'selected_sha256':[r['image_sha256'] for r in selected],'selected_before_GT_download':True,'GT_pixels_read':False})
        obs=[{'sample_id':'iit_'+r['source_image_id'],'image_path':r['image_path'],'action':'hold'} for r in selected]
        jsonl(out/'observations_ready.jsonl',obs)
        print('RGB_READY',flush=True)
        def gt(row):
            identifier=row['source_image_id']; member=f'IIT_Affordances_2017/affordances_labels/{identifier}.txt'
            path=labels/(identifier+'.txt')
            if not path.exists(): path.write_bytes(source.member(member))
            return path,sha(path.read_bytes()),member
        with ThreadPoolExecutor(max_workers=4) as pool: truths=list(pool.map(gt,selected))
        samples=[{**r,'sample_id':'iit_'+r['source_image_id'],'dataset':'IIT-AFF','action':'hold','split':'evaluation','evidence_role':'fresh_image_cross_source_Development','native_label_txt':str(path),'native_label_txt_sha256':digest,'gt_path':str(path),'gt_sha256':digest,'source_GT_member':member,'label_kind':'native_discrete_affordance_mask','target_label_ids':[5,9],'target_scope':'full-image all-instance union; no object-box clipping','primary':True} for r,(path,digest,member) in zip(selected,truths)]
        jsonl(out/'samples.jsonl',samples)
        shutil.copyfile(out/'observations_ready.jsonl',out/'observations.jsonl')
        dump(out/'data_summary.json',{'count':len(samples),'excluded_old':len(banned),'RGB_checked':len(inspected),'RGB_downloaded':len(list(images.glob('*.jpg'))),'duplicates':len(rejected),'objects':dict(Counter(r['object'] for r in samples)),'source_prefixes':dict(Counter(r['source_image_id'].split('_')[0] for r in samples)),'GT_pixels_read':False,'GT_crc_sha_only':True,'samples_sha256':sha((out/'samples.jsonl').read_bytes()),'observations_sha256':sha((out/'observations.jsonl').read_bytes())})
        status='SUCCESS'
    finally:
        dump(out/'acquisition_receipt.json',{'status':status,'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'cpu_scope':'one process including all threads','network_wall_s':time.monotonic()-network_start if 'network_start' in locals() else 0,'network_received_body_bytes':budget['bytes'],'network_reserved_upper_bound_bytes':budget['reserved']+500000,'download_limit_bytes':300000000,'network_wall_limit_s':900,'GT_pixels_read':False,'requests':source.receipts})
        print(status,flush=True)

if __name__=='__main__': main()
