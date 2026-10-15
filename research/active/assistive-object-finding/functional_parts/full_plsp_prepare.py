"""Fetch official PLSP training dependencies; never fetch AGD test pixel/GT payloads.

Resumable exact-range chunks, explicit full Seen training membership, CRC verification.
This is data preparation, not a smaller training substitute.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, html, io, json, os, re, struct, time, zlib, zipfile
from collections import defaultdict, Counter
from pathlib import Path
import requests

AGD_ID = '1OEz25-u1uqKfeuyCqy7hmiOv7lIWfigk'
EXTRA_ID = '1m0A1ke7n2aplJXFqLJtV2vcDtrpKlCLc'
CLIP_URL = 'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt'

def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')

def drive_form(file_id):
    r=requests.get('https://drive.google.com/uc',params={'id':file_id,'export':'download'},timeout=30)
    r.raise_for_status()
    m=re.search(r'<form[^>]+action="([^"]+)"',r.text)
    if not m: raise RuntimeError('Official Drive download form not available')
    return html.unescape(m.group(1)),dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',r.text))

def get_range(url,params,lo,hi,total=None):
    for attempt in range(4):
        try:
            with requests.get(url,params=params,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(20,60)) as r:
                r.raise_for_status()
                cr=r.headers.get('Content-Range','')
                if r.status_code!=206 or not cr.startswith(f'bytes {lo}-{hi}/'): raise RuntimeError(f'Range refused {r.status_code}: {cr}')
                size=int(cr.split('/')[-1])
                if total is not None and size!=total: raise RuntimeError('Archive size changed')
                data=r.content
                if len(data)!=hi-lo+1: raise RuntimeError('Truncated range')
                return data,size
        except requests.RequestException:
            if attempt==3: raise
            time.sleep(1+attempt)

def fetch_chunks(url,params,start,end,total,folder,workers=6):
    folder.mkdir(exist_ok=True,parents=True)
    chunk=16*1024*1024
    jobs=[(lo,min(lo+chunk,end)) for lo in range(start,end,chunk)]
    def fetch(job):
        lo,stop=job; p=folder/f'{lo}_{stop}.bin'
        receipt=p.with_suffix('.json')
        if p.exists() and receipt.exists() and p.stat().st_size==stop-lo:
            meta=json.loads(receipt.read_text())
            if hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']: return 0
        data,_=get_range(url,params,lo,stop-1,total)
        tmp=p.with_suffix('.part');tmp.write_bytes(data);tmp.replace(p)
        dump(receipt,{'lo':lo,'end':stop,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        return len(data)
    new=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for n,b in enumerate(pool.map(fetch,jobs),1):
            new+=b
            if n%10==0 or n==len(jobs): print(json.dumps({'chunks':n,'total_chunks':len(jobs),'new_bytes':new}),flush=True)
    return jobs,new

class ChunkReader:
    def __init__(self,folder,jobs): self.folder=folder;self.jobs=jobs;self.cached=None
    def read(self,offset,n):
        parts=[]
        while n:
            lo,end=next((lo,end) for lo,end in self.jobs if lo<=offset<end)
            count=min(n,end-offset)
            with (self.folder/f'{lo}_{end}.bin').open('rb') as f:
                f.seek(offset-lo);parts.append(f.read(count))
            offset+=count;n-=count
        return b''.join(parts)

def data(args):
    root=args.out;root.mkdir(exist_ok=True,parents=True)
    all_entries=sorted(json.loads(args.index.read_text()),key=lambda x:x['offset'])
    selected=[e for e in all_entries if e['name'].startswith('AGD20K/Seen/trainset/') and e['size']]
    start=selected[0]['offset'];last=selected[-1]['offset']
    end=next(e['offset'] for e in all_entries if e['offset']>last)
    intervening=[e for e in all_entries if start<=e['offset']<end]
    assert all(e['name'].startswith('AGD20K/Seen/trainset/') for e in intervening)
    assert all('/GT/' not in e['name'] and '/testset/' not in e['name'] for e in intervening)
    manifest={'start':start,'end':end,'bytes':end-start,'files':len(selected),'archive_bytes':6702534286,'selection':'every nonempty Seen/trainset member; no testset or GT byte payload','entries':selected}
    dump(root/'train_manifest.json',manifest)
    url,params=drive_form(AGD_ID)
    jobs,new=fetch_chunks(url,params,start,end,6702534286,root/'rgb_chunks',args.workers)
    reader=ChunkReader(root/'rgb_chunks',jobs)
    done=[]
    for i,e in enumerate(selected):
        p=root/'data'/e['name'];p.parent.mkdir(exist_ok=True,parents=True)
        if p.exists() and p.stat().st_size==e['size'] and f'{zlib.crc32(p.read_bytes()):08x}'==e['crc32']:
            done.append({'name':e['name'],'bytes':e['size'],'crc32':e['crc32']});continue
        h=reader.read(e['offset'],30);fields=struct.unpack('<4s5H3L2H',h);assert fields[0]==b'PK\x03\x04'
        namelen,extra=fields[-2:]
        assert reader.read(e['offset']+30,namelen).decode()==e['name']
        packed=reader.read(e['offset']+30+namelen+extra,e['compressed_size'])
        b=zlib.decompress(packed,-15) if fields[3]==8 else packed
        assert len(b)==e['size'] and f'{zlib.crc32(b):08x}'==e['crc32']
        p.write_bytes(b);done.append({'name':e['name'],'bytes':len(b),'crc32':e['crc32']})
        if (i+1)%2000==0: print(json.dumps({'extracted':i+1,'total':len(selected)}),flush=True)
    dump(root/'data_receipt.json',{'status':'complete','files':len(done),'new_bytes':new,'range_bytes':end-start,'no_test_GT':True,'verified':'CRC32 and length of every official ZIP entry'})

def assets(args):
    root=args.out;root.mkdir(exist_ok=True,parents=True)
    url,params=drive_form(EXTRA_ID)
    _,total=get_range(url,params,0,0)
    tail,_=get_range(url,params,max(0,total-65557),total-1,total)
    pos=tail.rfind(b'PK\x05\x06');eocd=struct.unpack_from('<4s4H2LH',tail,pos)
    cdsize,cdoffset=eocd[5:7]
    cd,_=get_range(url,params,cdoffset,cdoffset+cdsize-1,total)
    names=[];pos=0
    while pos<len(cd):
        fields=struct.unpack_from('<4s6H3L5H2L',cd,pos);assert fields[0]==b'PK\x01\x02'
        nl,xl,cl=fields[10:13];name=cd[pos+46:pos+46+nl].decode();names.append(name);pos+=46+nl+xl+cl
    dump(root/'extra_manifest.json',{'bytes':total,'entries':names,'file_id':EXTRA_ID})
    assert not any('/GT/' in n or '/testset/' in n for n in names), 'Do not fetch evaluation payload'
    jobs,new=fetch_chunks(url,params,0,total,total,root/'extra_chunks',args.workers)
    archive=root/'data.zip'
    with archive.open('wb') as f:
        for lo,end in jobs:f.write((root/'extra_chunks'/f'{lo}_{end}.bin').read_bytes())
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p=root/'data'/info.filename
            assert p.resolve().is_relative_to((root/'data').resolve())
            if not info.is_dir():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(info))
    clip=root/'ViT-B-16.pt'
    expected=CLIP_URL.split('/')[-2]
    if not clip.exists() or hashlib.sha256(clip.read_bytes()).hexdigest()!=expected:
        with requests.get(CLIP_URL,stream=True,timeout=(20,60)) as r:
            r.raise_for_status();n=0
            with clip.with_suffix('.part').open('wb') as f:
                for b in r.iter_content(1024*1024):
                    n+=len(b);assert n<500*1024*1024;f.write(b)
            clip.with_suffix('.part').replace(clip)
        assert hashlib.sha256(clip.read_bytes()).hexdigest()==expected
    dump(root/'assets_receipt.json',{'extra_bytes':total,'extra_new_bytes':new,'extra_entries':len(names),'clip_bytes':clip.stat().st_size,'clip_sha256':expected,'status':'complete','no_test_GT':True})

def audit(args):
    root=args.out
    receipt=json.loads((root/'data_receipt.json').read_text())
    assert receipt['status']=='complete'
    entries=json.loads((root/'train_manifest.json').read_text())['entries']
    def one(e):
        p=root/'data'/e['name'];b=p.read_bytes()
        assert len(b)==e['size'] and f'{zlib.crc32(b):08x}'==e['crc32']
        view,action,obj,filename=e['name'].split('/')[3:]
        return {'name':e['name'],'view':view,'action':action,'object':obj,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers,4)) as pool: rows=list(pool.map(one,entries))
    groups=defaultdict(list)
    for row in rows: groups[row['sha256']].append(row)
    views={v:[r for r in rows if r['view']==v] for v in ['egocentric','exocentric']}
    report={'status':'PASS','file_rows':len(rows),'unique_rgb_sha256':len(groups),'by_view':{v:{'file_rows':len(rr),'unique_rgb_sha256':len({r['sha256'] for r in rr})} for v,rr in views.items()},'cross_action_same_rgb_groups':sum(len({r['action'] for r in rr})>1 for rr in groups.values()),'cross_view_same_rgb_groups':sum(len({r['view'] for r in rr})>1 for rr in groups.values()),'duplicate_rgb_groups':sum(len(rr)>1 for rr in groups.values()),'action_object_pairs':len({(r['action'],r['object']) for r in rows}),'no_test_GT':True,'independence_limit':'SHA256 verifies exact RGB bytes only, not near-duplicate photographic independence'}
    (root/'training_rgb_content_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    dump(root/'training_rgb_content_audit.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['data','assets','audit']);p.add_argument('--out',type=Path,required=True);p.add_argument('--index',type=Path);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    started=time.time();cpu=time.process_time()
    try:globals()[a.mode](a)
    finally: print(json.dumps({'mode':a.mode,'wall_s':time.time()-started,'cpu_s':time.process_time()-cpu}),flush=True)
