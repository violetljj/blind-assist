"""Bounded official weight acquisition; immutable revision and local-only receipt."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import requests

def acquire(root, model_id, filename, cap, timeout, metadata=None, curl=False):
    start=time.perf_counter(); root=Path(root); root.mkdir(parents=True,exist_ok=True)
    ledger=root/'download_ledger.json'
    old=json.loads(ledger.read_text()) if ledger.exists() else {'transferred_bytes':0,'attempts':[]}
    if metadata is not None:meta=json.loads(Path(metadata).read_text('utf-8-sig'))
    else:
        response=requests.get(f'https://huggingface.co/api/models/{model_id}?blobs=true',timeout=30);response.raise_for_status();meta=response.json()
    rev=meta['sha']; sibling=next(x for x in meta['siblings'] if x['rfilename']==filename)
    size=sibling.get('size',sibling.get('lfs',{}).get('size')); expected=sibling.get('lfs',{}).get('sha256')
    target=root/model_id.replace('/','--');target.mkdir(exist_ok=True)
    (target/'official_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    url=f'https://huggingface.co/{model_id}/resolve/{rev}/{filename}'
    path=target/filename;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve existing acquisition, no implicit redownload')
    if size is None or old['transferred_bytes']+size>cap:raise RuntimeError('Download cap reservation failed')
    rec={'model_id':model_id,'filename':filename,'revision':rev,'url':url,'advertised_bytes':size,'advertised_sha256':expected,'bytes':0,'status':'INCOMPLETE'}
    old['attempts'].append(rec)
    try:
        h=hashlib.sha256()
        if curl:
            result=subprocess.run(['C:/Windows/System32/curl.exe','--fail','--location','--silent','--show-error','--max-time',str(timeout),'--max-filesize',str(size),'--output',str(path),'--write-out','%{size_download}',url],capture_output=True,text=True)
            rec['backend']='Windows curl Schannel; official cached metadata'
            rec['bytes']=int(float(result.stdout or 0));old['transferred_bytes']+=rec['bytes']
            if result.returncode:raise RuntimeError(result.stderr)
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
        else:
            with requests.get(url,stream=True,timeout=(30,40)) as response:
                response.raise_for_status()
                with path.open('xb') as stream:
                    for chunk in response.iter_content(4*1024*1024):
                        if not chunk:continue
                        if time.perf_counter()-start>timeout or old['transferred_bytes']+len(chunk)>cap:raise TimeoutError('Acquisition allocation reached')
                        stream.write(chunk);h.update(chunk);rec['bytes']+=len(chunk);old['transferred_bytes']+=len(chunk)
                        ledger.write_text(json.dumps(old,indent=2),encoding='utf-8')
        rec['sha256']=h.hexdigest()
        assert rec['bytes']==size and (expected is None or expected==rec['sha256'])
        rec['status']='COMPLETE'
    except Exception as exc:
        rec['status']='FAILED';rec['error']=repr(exc);raise
    finally:
        rec['command_wall_s']=time.perf_counter()-start
        ledger.write_text(json.dumps(old,indent=2),encoding='utf-8')
        (target/'weight_receipt.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
        print(json.dumps(rec),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model-id',required=True);p.add_argument('--filename',required=True);p.add_argument('--cap',type=int,default=7800000000);p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--metadata');p.add_argument('--curl',action='store_true')
    a=p.parse_args();acquire(a.root,a.model_id,a.filename,a.cap,a.timeout,a.metadata,a.curl)
