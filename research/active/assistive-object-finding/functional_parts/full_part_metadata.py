"""Metadata-only census; never decodes test RGB or test masks."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import time
import zipfile
import re
import gc
import io
import tarfile
import zlib
import requests

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/functional-parts-full-20261015/metadata'
PACO_URL = 'https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip'
PACO_SHA = '02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80'

def save(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def acquire():
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = []
    s = requests.Session()
    s.trust_env = False
    p = OUT / 'paco_lvis_v1.zip'
    if not p.exists():
        start = time.monotonic()
        with s.get(PACO_URL, stream=True, timeout=(30, 90)) as r:
            r.raise_for_status()
            assert int(r.headers['Content-Length']) < 2 * 1024**3
            with p.open('wb') as f:
                for b in r.iter_content(1024*1024):
                    f.write(b)
        receipts.append(dict(url=PACO_URL, bytes=p.stat().st_size, seconds=time.monotonic()-start))
    assert digest(p) == PACO_SHA, 'Official checksum mismatch; preserve partial for inspection'
    receipts.append(dict(path=str(p), sha256=digest(p), official_sha_match=True))
    with zipfile.ZipFile(p) as z:
        save('paco_archive_index.json', [dict(name=i.filename, bytes=i.file_size) for i in z.infolist()])
    url = 'https://huggingface.co/api/datasets/zifuwan/InstructPart/tree/main?recursive=true&limit=1000'
    tree=[]
    page=0
    while url:
        p = OUT / f'instructpart_tree_{page:02d}.json'
        h = OUT / f'instructpart_tree_{page:02d}_next.json'
        if p.exists() and h.exists():
            data=json.loads(p.read_text(encoding='utf8'))
            nxt=json.loads(h.read_text())['next']
        else:
            r=requests.get(url, timeout=60)
            r.raise_for_status()
            data=r.json()
            nxt=r.links.get('next',{}).get('url')
            save(p.name,data)
            save(h.name,dict(next=nxt))
            receipts.append(dict(url=url, bytes=len(r.content)))
        tree.extend(data)
        url=nxt
        page+=1
    save('instructpart_full_tree.json',tree)
    save('acquisition_receipt.json',receipts)
    print(json.dumps(dict(paco_bytes=(OUT/'paco_lvis_v1.zip').stat().st_size,hf_entries=len(tree),hf_json=[x['path'] for x in tree if x['path'].endswith('.json')]),ensure_ascii=False),flush=True)

FUNCTIONAL_OBJECTS = {
    'tools': ['knife','scissors','hammer','wrench','pliers','screwdriver','drill','broom'],
    'containers_tableware': ['bottle','bowl','can','carton','cup','glass_(drink_container)','jar','mug','pan_(for_cooking)','plate','box','bucket','crate','kettle','spoon','tray','vase','basket'],
    'appliances_controls': ['blender','fan','microwave_oven','lamp','remote_control','cellular_telephone','laptop_computer','mouse_(computer_equipment)','telephone','television_set','calculator'],
    'writing': ['pen','pencil'],
}

def paco_census():
    outputs={}
    with zipfile.ZipFile(OUT/'paco_lvis_v1.zip') as z:
        for split in ['val','train']:
            data=json.loads(z.read(f'paco_lvis_v1_{split}.json'))
            cats={c['id']:c for c in data['categories']}
            objcats={i:c for i,c in cats.items() if c['supercategory']=='OBJECT'}
            objects={a['id']:a for a in data['annotations'] if a['category_id'] in objcats}
            children=collections.defaultdict(list)
            orphans=[]
            partcounts=collections.Counter()
            for a in data['annotations']:
                if a['category_id'] in objcats:
                    continue
                name=cats[a['category_id']]['name']
                partcounts[name]+=1
                if a['obj_ann_id'] not in objects:
                    orphans.append(a['id'])
                else:
                    children[a['obj_ann_id']].append((a['id'],name,a['image_id']))
            object_counts=collections.Counter(objcats[a['category_id']]['name'] for a in objects.values())
            pairs=collections.Counter()
            per_object=[]
            for aid,child in children.items():
                names=sorted(set(n.split(':')[-1] for _,n,_ in child))
                objname=objcats[objects[aid]['category_id']]['name']
                for i,x in enumerate(names):
                    for y in names[i+1:]:
                        pairs[f'{objname}|{x}|{y}']+=1
                per_object.append(dict(object_annotation_id=aid,image_id=objects[aid]['image_id'],object=objname,parts=names,part_annotation_ids=[x[0] for x in child]))
            selection={g:set(ns) for g,ns in FUNCTIONAL_OBJECTS.items()}
            groups={}
            for g,names in selection.items():
                selected={a['id']:a for a in objects.values() if objcats[a['category_id']]['name'] in names}
                groups[g]=dict(exact_category_names=sorted(names),missing_category_names=sorted(names-set(object_counts)),images=len({a['image_id'] for a in selected.values()}),objects=len(selected),parts=sum(len(children[i]) for i in selected),objects_with_2plus_parts=sum(len(children[i])>=2 for i in selected))
            requested=set().union(*selection.values())
            selected={a['id']:a for a in objects.values() if objcats[a['category_id']]['name'] in requested}
            out=dict(split=split,images=len(data['images']),unique_image_ids=len({i['id'] for i in data['images']}),object_annotations=len(objects),part_annotations=sum(partcounts.values()),object_categories=len(objcats),observed_object_categories=len(object_counts),part_combination_categories=len(cats)-len(objcats),observed_part_combination_categories=len(partcounts),parent_links_valid=sum(map(len,children.values())),orphan_part_ids=orphans,parent_image_mismatches=sum(im!=objects[aid]['image_id'] for aid,ch in children.items() for _,_,im in ch),objects_with_any_parts=len(children),objects_with_2plus_part_annotations=sum(len(v)>=2 for v in children.values()),objects_with_2plus_distinct_part_names=sum(len(set(n for _,n,_ in v))>=2 for v in children.values()),object_counts=dict(object_counts),part_counts=dict(partcounts),part_pairs=dict(pairs),functional_groups=groups,functional_union=dict(images=len({a['image_id'] for a in selected.values()}),objects=len(selected),parts=sum(len(children[i]) for i in selected),objects_with_2plus_parts=sum(len(children[i])>=2 for i in selected)),non_exhaustive=dict(images_with_nonexhaustive_categories=sum(bool(i.get('not_exhaustive_category_ids')) for i in data['images']),nonexhaustive_category_entries=sum(len(i.get('not_exhaustive_category_ids',[])) for i in data['images']),images_with_explicit_negative_categories=sum(bool(i.get('neg_category_ids')) for i in data['images']),images_with_nonexhaustive_attributes=sum(bool(i.get('not_exhaustive_category_ids_attrs')) for i in data['images'])),role_labels_present=False,action_labels_present=False)
            save(f'paco_{split}_census.json',out)
            save(f'paco_{split}_parent_part_manifest.json',per_object)
            save(f'paco_{split}_categories.json',data['categories'])
            outputs[split]={k:v for k,v in out.items() if k not in ['object_counts','part_counts','part_pairs','functional_groups']}
            print(json.dumps(outputs[split],ensure_ascii=False),flush=True)
            del data,cats,objcats,objects,children,per_object
            gc.collect()
    save('paco_summary.json',outputs)

def drive_list(folder):
    p=OUT/f'drive_{folder}.html'
    if not p.exists():
        r=requests.get('https://drive.google.com/drive/folders/'+folder,timeout=60)
        r.raise_for_status()
        p.write_text(r.text,encoding='utf8')
    t=p.read_text(encoding='utf8')
    entries=[]
    for m in re.finditer(r'<tr data-selectable data-id="([^"]+)"(.*?)</tr>',t):
        labels=re.findall(r'data-tooltip="([^"]+)',m.group(2))
        entries.append(dict(id=m.group(1),labels=labels))
    save(f'drive_{folder}_entries.json',entries)
    print(json.dumps(entries,ensure_ascii=False),flush=True)

def instruct_census():
    tree=json.loads((OUT/'instructpart_full_tree.json').read_text(encoding='utf8'))
    bydir=collections.defaultdict(list)
    for x in tree:
        if x['type']=='file': bydir[str(pathlib.PurePosixPath(x['path']).parent)].append(x)
    outputs={}
    for directory,items in bydir.items():
        if not directory.endswith(('/images','/masks')): continue
        bybase=collections.defaultdict(set)
        byhash=collections.defaultdict(list)
        combos=collections.Counter()
        malformed=[]
        for x in items:
            stem=pathlib.PurePosixPath(x['path']).stem
            bits=stem.rsplit('-',2)
            if len(bits)!=3:
                malformed.append(stem)
                continue
            base,obj,part=bits
            bybase[base].add((obj,part))
            combos[f'{obj}|{part}']+=1
            sha=x.get('lfs',{}).get('oid')
            if sha: byhash[sha].append(x['path'])
        out=dict(files=len(items),total_payload_bytes=sum(x['size'] for x in items),filename_base_ids=len(bybase),base_ids_with_2plus_object_part_labels=sum(len(v)>=2 for v in bybase.values()),filename_derived_object_part_counts=dict(combos),lfs_sha256_present=sum(len(v) for v in byhash.values()),unique_lfs_sha256=len(byhash),sha256_groups_with_multiple_paths=sum(len(v)>1 for v in byhash.values()),malformed_filenames=malformed,action_and_role_labels='UNAVAILABLE: no annotation JSON in complete official HF tree')
        outputs[directory]=out
        save('instructpart_'+directory.replace('/','_')+'_base_groups.json',{k:sorted(v) for k,v in bybase.items()})
    train=[x for x in tree if '/train/train1800/images/' in x['path']]
    test=[x for x in tree if '/test/images/' in x['path']]
    th={x.get('lfs',{}).get('oid') for x in train}-{None}
    eh={x.get('lfs',{}).get('oid') for x in test}-{None}
    common=th&eh
    overlap=dict(exact_rgb_sha256_overlap_groups=len(common),train_paths_in_overlap=[x['path'] for x in train if x.get('lfs',{}).get('oid') in common],test_paths_in_overlap=[x['path'] for x in test if x.get('lfs',{}).get('oid') in common])
    save('instructpart_hf_census.json',dict(directory_counts=outputs,train1800_test_overlap=overlap,hf_release_status='INCOMPLETE_RELATIVE_TO_README',test_pixels_read=False,mask_pixels_read=False,notes=['Counts are complete for accessible official HF release, not full advertised InstructPart dataset.','Filename base IDs and LFS hashes establish source-image grouping, not object instance IDs.','Absent masks are not absent functional regions.']))
    print(json.dumps(dict(directories={k:{a:b for a,b in v.items() if a not in ['filename_derived_object_part_counts','malformed_filenames']} for k,v in outputs.items()},overlap_groups=len(common)),ensure_ascii=False),flush=True)

def umd_prefix():
    urls={k:f'https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-{k}.tar.gz' for k in ['tools','clutter']}
    receipt={}
    for kind,url in urls.items():
        p=OUT/f'umd_{kind}_prefix_64KiB.bin'
        previous=OUT/f'umd_{kind}_prefix_8MiB.bin'
        if previous.exists() and previous.stat().st_size:
            p=previous
        if not p.exists():
            with requests.get(url,headers={'Range':'bytes=0-65535'},stream=True,timeout=(20,60)) as r:
                r.raise_for_status()
                assert r.status_code==206 and int(r.headers['Content-Length'])<=65536
                save(f'umd_{kind}_range_headers.json',dict(r.headers))
                with p.open('wb') as f:
                    for b in r.iter_content(16384): f.write(b)
        content=p.read_bytes()
        raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(content) if content[:2]==b'\x1f\x8b' else content
        rows=[]
        offset=0
        while offset+512<=len(raw):
            header=raw[offset:offset+512]
            if not any(header): break
            info=tarfile.TarInfo.frombuf(header,'utf8','replace')
            rows.append(dict(name=info.name,size=info.size,type=info.type.decode('ascii','replace')))
            body=raw[offset+512:offset+512+info.size]
            if info.name.endswith('.txt') and len(body)==info.size:
                # Only plaintext split/document metadata; never mask or RGB payload.
                dest=OUT/('umd_'+kind+'_'+pathlib.PurePosixPath(info.name).name)
                dest.write_bytes(body)
            offset+=512+((info.size+511)//512)*512
        save(f'umd_{kind}_partial_index.json',rows)
        receipt[kind]=dict(downloaded_bytes=len(content),sha256=digest(p),is_gzip=content[:2]==b'\x1f\x8b',decompressed_prefix_bytes=len(raw),partial_tar_entries=len(rows),full_archive_census=False,headers=json.loads((OUT/f'umd_{kind}_range_headers.json').read_text()))
        print(kind,json.dumps(rows[:20]),flush=True)
    save('umd_range_receipt.json',receipt)
    lines=(OUT/'umd_tools_category_10_fold.txt').read_text().splitlines()
    folds=[(re.search(r'object: (\S+)',x).group(1),list(map(int,re.search(r'folds: (.*?)    object:',x).group(1).split()))) for x in lines if x.strip()]
    novel=[x.split() for x in (OUT/'umd_tools_novel_split.txt').read_text().splitlines() if x.strip()]
    save('umd_split_census.json',dict(tool_instances=len(folds),unique_tool_instances=len({a for a,_ in folds}),native_tool_category_counts=dict(collections.Counter(a.rsplit('_',1)[0] for a,_ in folds)),ten_fold_test_counts=[sum(v[i] for _,v in folds) for i in range(10)],each_tool_in_exactly_one_test_fold=all(sum(v)==1 for _,v in folds),novel_split_native_value_counts=dict(collections.Counter(a for a,b in novel)),novel_split_native_value_categories={k:dict(collections.Counter(b.rsplit('_',1)[0] for a,b in novel if a==k)) for k in sorted({a for a,b in novel})},novel_split_value_semantics='Not documented in extracted split; do not assume 1=train or test',frame_counts='UNKNOWN: gzip prefix is not full archive index',manual_frame_exact_ids='UNKNOWN: README every-third does not identify index residue'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['acquire','paco','drive','instruct','umd'])
    parser.add_argument('--folder')
    args=parser.parse_args()
    if args.mode=='acquire': acquire()
    elif args.mode=='paco': paco_census()
    elif args.mode=='instruct': instruct_census()
    elif args.mode=='umd': umd_prefix()
    else: drive_list(args.folder)
