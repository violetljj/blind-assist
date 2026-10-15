"""Fixed image-group functional-part Development pilot; no predictions read.

AGD continuous heatmaps remain heatmaps. IIT hold is the full-image union of
native semantic IDs 5 and 9, including every instance; boxes are evaluator-only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import io
import json
from pathlib import Path
import re
import struct
import threading
import time
import zlib

import numpy as np
from PIL import Image
import requests

SEED = "20261015-fp"
CONFIG = {"AGD20K": ("agd20k_audit", "1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk", 6702534286),
          "IIT-AFF": ("iit_aff_audit", "15B0dqmqyKaWRWgZ9xP1lm0tfGUpyRVp3", 5448607854)}
OBJECTS = {0:"bowl",1:"tvm",2:"pan",3:"hammer",4:"knife",5:"cup",6:"drill",7:"racket",8:"spatula",9:"bottle"}


def sha(b): return hashlib.sha256(b).hexdigest()
def rank(x): return sha((str(x)+SEED).encode())
def dump(p, d): p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
def jsonl(p, rows): p.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


class Source:
    def __init__(self, dataset, old, out, deadline, budget):
        audit, self.file_id, self.size = CONFIG[dataset]
        self.dataset, self.old, self.out = dataset, old/audit, out
        self.entries = {e["name"]: e for e in json.loads((self.old/"zip_entries.json").read_text(encoding="utf-8"))}
        self.deadline, self.budget, self.cache = deadline, budget, []
        self.lock, self.form = budget["lock"], None
        self.receipts = []
        for path in sorted(self.old.glob("range_*.bin")):
            lo, hi = map(int, path.stem.split("_")[1:])
            self.cache.append((lo, hi+1, path.read_bytes()))
        tail = (self.old/"zip_tail.bin").read_bytes()
        self.cache.append((self.size-len(tail), self.size, tail))

    def cached(self, lo, hi):
        for start, end, data in self.cache:
            if start <= lo and hi < end: return data[lo-start:hi-start+1]
        return None

    def connect(self):
        with requests.get("https://drive.google.com/uc", params={"id": self.file_id,"export":"download"}, timeout=(10,20), stream=True) as response:
            response.raise_for_status()
            body=bytearray()
            for block in response.iter_content(8192):
                body.extend(block)
                if len(body)>500000:raise ValueError("Unexpected public form body; no full archive download")
        text=body.decode("utf-8")
        match = re.search(r'<form[^>]+action="([^"]+)"', text)
        if not match: raise ValueError("Public download form unavailable; no whole-body fallback")
        self.form = (html.unescape(match.group(1)), dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text)))
        self.budget["bytes"] += len(body)

    def get(self, lo, hi, offline=False):
        cached = self.cached(lo, hi)
        if cached is not None: return cached
        if offline: raise ValueError(f"Required metadata range absent from old cache: {lo}-{hi}")
        if self.form is None: raise RuntimeError("Connect source before downloads")
        if time.monotonic() > self.deadline: raise TimeoutError("Acquisition deadline reached")
        if self.budget["bytes"] + hi-lo+1 > 750_000_000: raise RuntimeError("750 MB download budget reached")
        url, params = self.form
        for attempt in range(3):
            try:
                with requests.get(url, params=params, headers={"Range":f"bytes={lo}-{hi}"}, stream=True, timeout=(10,25)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {lo}-{hi}/{self.size}":
                        raise ValueError("Exact HTTP range unsupported; no full-body fallback")
                    body = bytearray()
                    for block in response.iter_content(262144):
                        with self.lock:
                            self.budget["bytes"] += len(block)
                        body.extend(block)
                        if len(body)>hi-lo+1 or self.budget["bytes"]>750_000_000 or time.monotonic()>self.deadline:
                            raise TimeoutError("Download bound reached")
                assert len(body)==hi-lo+1
                body=bytes(body)
                with self.lock:
                    self.receipts.append({"archive_id":self.file_id,"range":f"bytes={lo}-{hi}","bytes":len(body),"sha256":sha(body),"status":206})
                return body
            except requests.RequestException:
                if attempt==2: raise
        raise RuntimeError("Unreachable")

    def member(self, name, offline=False):
        entry=self.entries[name];lo=entry["offset"]
        if offline:
            header=self.get(lo,lo+29,offline=True)
            fields=struct.unpack_from("<4s5H3L2H",header,0)
            n,e=fields[-2:]
            data=self.get(lo,lo+30+n+e+entry["compressed_size"]-1,offline=True)
        else:
        # One contiguous request per file; old metadata cache avoids all metadata traffic.
            hi=min(self.size-1,lo+30+len(name.encode())+entry["compressed_size"]+4096)
            data=self.get(lo,hi)
        fields=struct.unpack_from("<4s5H3L2H",data,0)
        assert fields[0]==b"PK\x03\x04"
        n,e=fields[-2:];start=30+n+e
        assert data[30:30+n].decode()==name
        packed=data[start:start+entry["compressed_size"]]
        plain=zlib.decompress(packed,-15) if fields[3]==8 else packed
        assert len(plain)==entry["size"] and f"{zlib.crc32(plain):08x}"==entry["crc32"]
        return plain


def agd_index(source):
    images, gt={},{}
    for name, entry in source.entries.items():
        p=name.split("/")
        if entry["size"] and len(p)==7 and p[:3]==["AGD20K","Seen","testset"]:
            kind,action,obj,filename=p[3:]
            if kind=="egocentric" and filename.endswith(".jpg"): images[(action,obj,Path(filename).stem)]=name
            if kind=="GT" and filename.endswith(".png"): gt[(action,obj,Path(filename).stem)]=name
    groups=defaultdict(list)
    for key in sorted(set(images)&set(gt)):
        a,o,identifier=key
        groups[identifier].append({"action":a,"object":o,"image_member":images[key],"gt_member":gt[key]})
    missing=[{"action":a,"object":o,"source_image_id":i,"image_member":images[(a,o,i)],"status":"NOT_EVALUABLE","reason":"Missing native GT; never absent"} for a,o,i in sorted(set(images)-set(gt))]
    return groups,missing


def choose_agd(groups, banned):
    available={i:pairs for i,pairs in groups.items() if i not in banned}
    actions=sorted({p["action"] for pairs in available.values() for p in pairs})
    edges={a:sorted([i for i,pairs in available.items() if any(p["action"]==a for p in pairs)],key=rank) for a in actions}
    slots=[(a,k) for a in sorted(actions,key=lambda a:(len(edges[a]),a)) for k in range(3)]
    assigned={}
    def augment(slot,seen):
        action,_=slot
        for identifier in edges[action]:
            if identifier in seen: continue
            seen.add(identifier)
            if identifier not in assigned or augment(assigned[identifier],seen):
                assigned[identifier]=slot;return True
        return False
    assert all(augment(slot,set()) for slot in slots),"Cannot cover 36 actions with three independent images"
    selected={i:next(p for p in available[i] if p["action"]==slot[0]) for i,slot in assigned.items()}
    ac=Counter(p["action"] for p in selected.values());oc=Counter(p["object"] for p in selected.values())
    while len(selected)<240:
        options=[(oc[p["object"]],ac[p["action"]],rank(i+p["action"]),i,p) for i,pairs in available.items() if i not in selected for p in pairs]
        _,_,_,identifier,p=min(options,key=lambda x:x[:3]);selected[identifier]=p;ac[p["action"]]+=1;oc[p["object"]]+=1
    by_action=defaultdict(list)
    for i,p in selected.items(): by_action[p["action"]].append(i)
    calibration,evaluation=set(),set()
    for action,ids in by_action.items():
        ids=sorted(ids,key=rank);calibration.add(ids[0]);evaluation.add(ids[1])
    remaining=sorted(set(selected)-calibration-evaluation,key=rank)
    calibration.update(remaining[:80-len(calibration)])
    evaluation=set(selected)-calibration
    assert len(calibration)==80 and len(evaluation)==160 and min(ac.values())>=3
    return [{"source_image_id":i,"primary":p,"pairs":available[i],"split":"calibration" if i in calibration else "evaluation"} for i,p in sorted(selected.items(),key=lambda kv:rank(kv[0]))]


def iit_index(source):
    native_test=(source.old/"test.txt").read_text().splitlines()
    native_test=[x.strip().removesuffix(".jpg") for x in native_test if x.strip()]
    boxes={}
    for identifier in native_test:
        name=f"IIT_Affordances_2017/object_labels/{identifier}.txt"
        text=source.member(name,offline=True).decode()
        boxes[identifier]=[[int(v) for v in line.split()] for line in text.splitlines() if line.strip()]
        assert all(len(row)==5 for row in boxes[identifier])
    return boxes


def choose_iit(boxes,banned):
    selected=[];used=set(banned)
    for label in sorted(OBJECTS,key=lambda k:(sum(any(r[0]==k for r in rows) for rows in boxes.values()),k)):
        candidates=sorted([i for i,rows in boxes.items() if i not in used and any(r[0]==label for r in rows)],key=rank)
        assert len(candidates)>=4
        for i in candidates[:4]: selected.append({"source_image_id":i,"object_id":label,"object":OBJECTS[label],"boxes":boxes[i]});used.add(i)
    two_cal=set(sorted(OBJECTS,key=lambda k:rank("iit-class"+str(k)))[:6])
    for label in OBJECTS:
        group=sorted([r for r in selected if r["object_id"]==label],key=lambda r:rank(r["source_image_id"]))
        for index,row in enumerate(group):row["split"]="calibration" if index<(2 if label in two_cal else 1) else "evaluation"
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--old",type=Path,required=True)
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--wall-seconds",type=int,default=600)
    parser.add_argument("--workers",type=int,default=12)
    args=parser.parse_args();started=time.monotonic();cpu=time.process_time();deadline=started+args.wall_seconds
    assert "artifacts.local" in args.root.as_posix()
    out=args.root.resolve()/"data";out.mkdir(parents=True,exist_ok=True);budget={"bytes":0,"lock":threading.Lock()}
    sources={d:Source(d,args.old.resolve(),out,deadline,budget) for d in CONFIG}
    old_agd=json.loads((sources["AGD20K"].old/"summary.json").read_text(encoding="utf-8"))["samples"]
    old_iit=json.loads((sources["IIT-AFF"].old/"summary.json").read_text(encoding="utf-8"))["samples"]
    banned_agd={r["key"][3] for r in old_agd};banned_iit={r["native_test_id"] for r in old_iit}
    old_hashes={r["image_sha256"] for r in old_agd}|{r["source_sha256"]["image"] for r in old_iit}
    groups,missing=agd_index(sources["AGD20K"]);boxes=iit_index(sources["IIT-AFF"])
    agd=choose_agd(groups,banned_agd);iit=choose_iit(boxes,banned_iit)
    jsonl(out/"agd_missing_gt.jsonl",missing)
    def plan():
        return {"seed":SEED,"evidence_role":"Development","AGD":{"unique_original_ids":len(groups),"selected":len(agd),"primary_actions":dict(Counter(r["primary"]["action"] for r in agd)),"primary_objects":dict(Counter(r["primary"]["object"] for r in agd)),"missing_GT":len(missing),"selected_groups":agd},"IIT":{"selected":len(iit),"objects":dict(Counter(r["object"] for r in iit)),"selected_groups":iit},"excluded_old_AGD_ids":sorted(banned_agd),"excluded_old_IIT_ids":sorted(banned_iit),"split_rule":"AGD: reserve one calibration and one evaluation primary group per action by SHA, then hash-fill calibration to80. IIT: hash-stratify class quotas16/24.","GT_selection":"No pixel values, GT areas or model predictions used"}
    dump(out/"selection_plan.json",plan());print(json.dumps({"stage":"plan","AGD_groups":len(agd),"actions":len({r['primary']['action'] for r in agd}),"minimum_action_count":min(Counter(r['primary']['action'] for r in agd).values()),"IIT":len(iit),"missing_GT":len(missing)}),flush=True)
    if args.plan_only:return
    for s in sources.values():s.connect()
    def save_member(dataset,name,kind,identifier):
        suffix=Path(name).suffix;path=out/dataset/kind/(identifier+suffix);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(sources[dataset].member(name))
        return path,path.read_bytes()
    replacements=[]
    while True:
        requests_to_get=[("AGD20K",r["primary"]["image_member"],"images",r["source_image_id"]) for r in agd]+[("IIT-AFF",f"IIT_Affordances_2017/rgb/{r['source_image_id']}.jpg","images",r["source_image_id"]) for r in iit]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:images=list(pool.map(lambda x:save_member(*x),requests_to_get))
        seen={h:"old_audit" for h in old_hashes};duplicates=[]
        for request,(path,body) in zip(requests_to_get,images):
            h=sha(body);dataset,_,_,identifier=request
            if h in seen:duplicates.append((dataset,identifier,seen[h]));replacements.append({"dataset":dataset,"dropped_id":identifier,"duplicate_of":seen[h],"image_sha256":h})
            else:seen[h]=identifier
        if not duplicates:break
        for dataset,identifier,_ in duplicates:
            (banned_agd if dataset=="AGD20K" else banned_iit).add(identifier)
        agd=choose_agd(groups,banned_agd);iit=choose_iit(boxes,banned_iit);dump(out/"selection_plan.json",plan())
    jobs=[]
    for row in agd:
        for pair in row["pairs"]:jobs.append(("AGD20K",pair["gt_member"],"GT",row["source_image_id"]+"__"+pair["action"]))
    for row in iit:
        identifier=row["source_image_id"]
        jobs.extend([("IIT-AFF",f"IIT_Affordances_2017/{folder}/{identifier}.txt",kind,identifier) for folder,kind in (("affordances_labels","native_labels"),("object_labels","boxes"))])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(lambda x:save_member(*x),jobs))
    samples,diagnostics=[] ,[]
    for row in agd:
        identifier=row["source_image_id"];image=out/"AGD20K/images"/(identifier+".jpg");body=image.read_bytes()
        with Image.open(image) as im:im.load();w,h=im.size
        for pair in row["pairs"]:
            gt=out/"AGD20K/GT"/(identifier+"__"+pair["action"]+".png")
            with Image.open(gt) as im:im.load();assert im.size==(w,h)
            r={"sample_id":f"agd_{identifier}__{pair['action']}","dataset":"AGD20K","source_image_id":identifier,"split":row["split"],"evidence_role":"Development","image_path":str(image),"image_sha256":sha(body),"width":w,"height":h,"gt_path":str(gt),"gt_sha256":sha(gt.read_bytes()),"action":pair["action"],"object":pair["object"],"label_kind":"native_continuous_heatmap","source_image_member":pair["image_member"],"source_GT_member":pair["gt_member"],"primary":pair==row["primary"]}
            (samples if r["primary"] else diagnostics).append(r)
    for row in iit:
        identifier=row["source_image_id"];image=out/"IIT-AFF/images"/(identifier+".jpg");body=image.read_bytes()
        with Image.open(image) as im:im.load();w,h=im.size
        text=out/"IIT-AFF/native_labels"/(identifier+".txt");native=np.loadtxt(text,dtype=np.uint8)
        assert native.shape==(h,w) and int(native.max())<=9
        gt=text.with_suffix(".png");Image.fromarray(native).save(gt)
        counts=Counter(native.ravel().tolist())
        samples.append({"sample_id":"iit_"+identifier,"dataset":"IIT-AFF","source_image_id":identifier,"split":row["split"],"evidence_role":"Development","image_path":str(image),"image_sha256":sha(body),"width":w,"height":h,"gt_path":str(gt),"gt_sha256":sha(gt.read_bytes()),"native_label_txt":str(text),"native_label_txt_sha256":sha(text.read_bytes()),"action":"hold","object":row["object"],"label_kind":"native_discrete_affordance_mask","target_label_ids":[5,9],"target_scope":"full-image union, all instances; no object-box clipping","target_5_pixels":counts[5],"target_9_pixels":counts[9],"native_empty":counts[5]+counts[9]==0,"object_boxes_native":row["boxes"],"box_convention":"obj_id,xmin,ymin,xmax,ymax; native inclusive coordinates; evaluator-only","primary":True})
    assert len(samples)==280 and len({r["image_sha256"] for r in samples})==280
    assert Counter((r["dataset"],r["split"]) for r in samples)==Counter({("AGD20K","calibration"):80,("AGD20K","evaluation"):160,("IIT-AFF","calibration"):16,("IIT-AFF","evaluation"):24})
    jsonl(out/"samples.jsonl",samples);jsonl(out/"diagnostic_samples.jsonl",diagnostics)
    jsonl(out/"agd_calibration.jsonl",[r for r in samples if r["dataset"]=="AGD20K" and r["split"]=="calibration"])
    jsonl(out/"observations.jsonl",[{k:r[k] for k in ("sample_id","image_path","action")} for r in samples])
    jsonl(out/"diagnostic_observations.jsonl",[{k:r[k] for k in ("sample_id","image_path","action")} for r in diagnostics])
    dump(out/"duplicate_replacements.json",replacements)
    for d,s in sources.items():dump(out/(d.replace('-','_')+"_range_receipts.json"),{"archive_id":s.file_id,"archive_size":s.size,"requests":s.receipts})
    summary={"status":"complete","independent_images":280,"main_queries":len(samples),"diagnostic_queries_same_images":len(diagnostics),"split_counts":{f"{d}_{s}":n for (d,s),n in Counter((r["dataset"],r["split"]) for r in samples).items()},"AGD_actions":len({r["action"] for r in samples if r["dataset"]=="AGD20K"}),"AGD_missing_native_GT":len(missing),"IIT_native_empty":sum(r.get("native_empty",False) for r in samples),"duplicate_replacements":len(replacements),"new_network_body_bytes":budget["bytes"],"wall_s":time.monotonic()-started,"cpu_s":time.process_time()-cpu,"backend":"CPU","placement_reason":"TASK_NOT_GPU_SUITABLE","samples_sha256":sha((out/"samples.jsonl").read_bytes()),"observations_sha256":sha((out/"observations.jsonl").read_bytes()),"limits":"Development only. AGD heatmap is not a binary mask. IIT hold is semantic IDs5/9 union, not safe contact/grasp success. Diagnostics share image denominator."}
    dump(out/"data_summary.json",summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
