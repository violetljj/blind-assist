"""Resumable bounded UMD archives and native filename/metadata census.

Does not decode RGB or label pixels. Extracts RGB, label.mat and text/code only.
The native manual-frame index remains UNKNOWN unless original evidence resolves it.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tarfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import requests

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'artifacts.local/work/functional-parts-full-20261015'
SOURCES={'tools':('https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-tools.tar.gz',2986470339),
         'clutter':('https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-clutter.tar.gz',186182704)}
NAMES={0:'background',1:'grasp',2:'cut',3:'scoop',4:'contain',5:'pound',6:'support',7:'wrap-grasp'}
IIT_MAP={0:0,1:5,2:2,3:None,4:1,5:7,6:8,7:9}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.new')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)


class Download:
    def __init__(self,out,budget,workers,chunk):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.budget=budget;self.workers=workers;self.chunk=chunk
        self.start=time.monotonic();self.state_path=self.out/'download_state.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'reserved_request_bytes':0,'completed_read_bytes':0,'completed_chunks':{},'prefixes':{},'requests':[],
            'budget_bytes':budget,'chunk_bytes':chunk,'resources':'single process, <=8 task-owned HTTP threads, no GPU'}
        if self.state['chunk_bytes']!=chunk:
            self.state.setdefault('chunk_refinements',[]).append({'from':self.state['chunk_bytes'],'to':chunk,
                'reason':'Observed proxy truncation after 1.5-2MB; smaller bounded range preserves download identity and budget'})
            self.state['chunk_bytes']=chunk
        if self.state['budget_bytes']!=budget:raise ValueError('Resume must preserve authorized budget')

    def event(self,obj):
        with (self.out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'elapsed_s':time.monotonic()-self.start,**obj})+'\n')

    def reserve(self,kind,start,end):
        with self.lock:
            charge=end-start+1
            auxiliary=self.out/'code_request_reservations.json'
            aux_reserved=json.loads(auxiliary.read_text())['reserved_bytes'] if auxiliary.exists() else 0
            if self.state['reserved_request_bytes']+charge+aux_reserved>self.budget:
                raise RuntimeError('Conservative request reservation would exceed authorized 3.6GB')
            self.state['reserved_request_bytes']+=charge
            ix=len(self.state['requests'])
            self.state['requests'].append({'archive':kind,'start':start,'end':end,'reserved_bytes':charge,'status':'IN_FLIGHT','known_read_bytes':0})
            save(self.state_path,self.state)
            return ix

    def prepare(self,kind):
        url,size=SOURCES[kind];target=self.out/f'{kind}.tar.gz'
        if not target.exists():
            with target.open('wb') as f:f.truncate(size)
        if target.stat().st_size!=size:raise ValueError('Archive destination size mismatch')
        if kind not in self.state['prefixes']:
            meta=BASE/'metadata'
            prefix=meta/('umd_tools_prefix_8MiB.bin' if kind=='tools' else 'umd_clutter_prefix_64KiB.bin')
            receipt=json.loads((meta/'umd_range_receipt.json').read_text())[kind]
            if prefix.stat().st_size!=receipt['downloaded_bytes'] or sha(prefix)!=receipt['sha256']:
                raise ValueError('Old prefix checksum mismatch')
            with target.open('r+b') as f:f.write(prefix.read_bytes());f.flush();os.fsync(f.fileno())
            self.state['prefixes'][kind]={'bytes':prefix.stat().st_size,'sha256':sha(prefix),'source':str(prefix),'reused_prior_task_bytes':True}
            save(self.state_path,self.state)
        prefix_bytes=self.state['prefixes'][kind]['bytes']
        coverage=[(0,prefix_bytes)]
        for key in self.state['completed_chunks']:
            k,a,b=key.split(':')
            if k==kind:coverage.append((int(a),int(b)+1))
        for request in self.state['requests']:
            if request['archive']==kind and request['status']=='FAILED' and request['known_read_bytes']:
                # Confirmed writes from an exception-closed stream are reusable.
                # Unknown bytes from interrupted streams are never admitted.
                coverage.append((request['start'],request['start']+request['known_read_bytes']))
        merged=[]
        for a,b in sorted(coverage):
            if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
            else:merged.append([a,b])
        self.state.setdefault('resume_coverage',{})[kind]=merged
        save(self.state_path,self.state)
        cursor=0
        gaps=[]
        for a,b in merged:
            if a>cursor:gaps.append((cursor,a))
            cursor=max(cursor,b)
        if cursor<size:gaps.append((cursor,size))
        for a,b in gaps:
            for start in range(a,b,self.chunk):
                end=min(b-1,start+self.chunk-1);yield kind,start,end,f'{kind}:{start}:{end}'

    def fetch(self,job):
        kind,start,end,key=job;url,size=SOURCES[kind];path=self.out/f'{kind}.tar.gz'
        error=None
        for attempt in range(2):
            ix=self.reserve(kind,start,end);received=0;begin=time.monotonic();h=hashlib.sha256()
            try:
                with requests.Session() as session:
                    # Existing official-prefix acquisition used the configured proxy.
                    # Direct connections timed out twice; preserve those receipts.
                    session.trust_env=True
                    with session.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(30,90)) as response:
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                            raise RuntimeError(f'Wrong range response: {response.status_code} {response.headers.get("Content-Range")}')
                        with path.open('r+b') as f:
                            f.seek(start)
                            for b in response.iter_content(256*1024):
                                received+=len(b)
                                if received>end-start+1:raise RuntimeError('Server exceeded reserved range')
                                f.write(b);h.update(b)
                            if received!=end-start+1:raise RuntimeError('Truncated range')
                            f.flush();os.fsync(f.fileno())
                with self.lock:
                    self.state['completed_read_bytes']+=received
                    self.state['requests'][ix].update(status='COMPLETE',known_read_bytes=received,seconds=time.monotonic()-begin)
                    self.state['completed_chunks'][key]={'sha256':h.hexdigest(),'bytes':received}
                    save(self.state_path,self.state)
                    done=len(self.state['completed_chunks'])
                    self.event({'stage':'download','completed_chunks':done,'reserved_bytes':self.state['reserved_request_bytes'],
                                'known_read_bytes':self.state['completed_read_bytes'],'archive':kind,'range_start':start})
                return
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                with self.lock:
                    self.state['completed_read_bytes']+=received
                    self.state['requests'][ix].update(status='FAILED',known_read_bytes=received,seconds=time.monotonic()-begin,error=error)
                    save(self.state_path,self.state)
                    self.event({'stage':'range_failure','archive':kind,'range_start':start,'attempt':attempt+1,'error':error})
        raise RuntimeError(f'Range failed twice; preserve partial and change diagnosis: {key}: {error}')

    def run(self):
        jobs=[]
        for kind in SOURCES:jobs.extend(self.prepare(kind))
        self.event({'stage':'download_start','pending_chunks':len(jobs),'workers':self.workers})
        errors=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            # At most eight submitted requests: failure cannot leave hundreds queued.
            iterator=iter(jobs);futures={}
            for _ in range(self.workers):
                job=next(iterator,None)
                if job:futures[pool.submit(self.fetch,job)]=job
            while futures:
                done,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    job=futures.pop(f)
                    try:f.result()
                    except Exception as exc:errors.append({'job':job,'error':str(exc)})
                    if not errors:
                        job=next(iterator,None)
                        if job:futures[pool.submit(self.fetch,job)]=job
        if errors:
            save(self.out/'terminal.json',{'status':'DOWNLOAD_INCOMPLETE','errors':errors,'state':str(self.state_path),'threads_closed':True})
            raise RuntimeError('Download incomplete; all HTTP worker threads closed; see terminal.json')
        receipts={}
        for kind,(url,size) in SOURCES.items():
            p=self.out/f'{kind}.tar.gz'
            receipts[kind]={'url':url,'bytes':size,'sha256':sha(p),'official_checksum_available':False}
        result={'status':'DOWNLOAD_COMPLETE','archives':receipts,'new_request_reserved_upper_bound_bytes':self.state['reserved_request_bytes'],
                'known_read_bytes':self.state['completed_read_bytes'],'prior_prefix_bytes':sum(x['bytes'] for x in self.state['prefixes'].values()),
                'wall_seconds':time.monotonic()-self.start,'threads_closed':True}
        save(self.out/'download_receipt.json',result);return result


def census(out):
    out=Path(out);receipt=json.loads((out/'download_receipt.json').read_text())
    frames=defaultdict(dict);entries=[];texts=[];counts=Counter();extracted_bytes=0
    for kind in SOURCES:
        archive=out/f'{kind}.tar.gz'
        if archive.stat().st_size!=receipt['archives'][kind]['bytes']:raise ValueError('Archive size changed')
        dest=(out/'extracted'/kind).resolve();dest.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                info={'archive':kind,'name':member.name,'size':member.size,'type':member.type.decode('ascii',errors='replace')}
                entries.append(info)
                if not member.isfile():continue
                p=PurePosixPath(member.name)
                if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
                name=p.name
                match=re.match(r'(.+?)_(rgb\.jpg|depth\.png|label\.mat|label_rank\.mat)$',name)
                if match:
                    stem,suffix=match.groups();group=p.parent.name;key=f'{kind}/{group}/{stem}'
                    if suffix in frames[key]:raise ValueError('Duplicate native frame/member')
                    frames[key][suffix]={'member':member.name,'size':member.size}
                    frames[key].update(archive=kind,group=group,frame_id=stem)
                    counts[kind+'/'+suffix]+=1
                keep=name.endswith('_rgb.jpg') or name.endswith('_label.mat') or p.suffix.lower() in ['.txt','.m','.json','.csv']
                if keep:
                    target=(dest/Path(*p.parts)).resolve()
                    if not target.is_relative_to(dest):raise ValueError('Escaping extraction root')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    source=tar.extractfile(member)
                    with target.open('wb') as f:
                        for b in iter(lambda:source.read(1024**2),b''):f.write(b)
                    extracted_bytes+=member.size
                    if match and match[2] in ['rgb.jpg','label.mat']:
                        frames[key][match[2]]['path']=str(target)
                    else:texts.append({'archive':kind,'member':member.name,'path':str(target)})
    with (out/'archive_index.jsonl').open('w',encoding='utf-8') as f:
        for x in entries:f.write(json.dumps(x)+'\n')
    with (out/'frame_manifest.jsonl').open('w',encoding='utf-8') as f:
        for key,x in sorted(frames.items()):f.write(json.dumps({'source_id':key,**x,'manual_annotation_status':'UNKNOWN'})+'\n')
    save(out/'native_text_files.json',texts)
    groups=defaultdict(Counter)
    for x in frames.values():groups[x['archive']][x['group']]+=1
    structure=[]
    # Author fold column 1==test; only probe a tool explicitly not in fold-1 test.
    fold=next((x for x in texts if Path(x['path']).name=='category_10_fold.txt'),None)
    if fold:
        train_tools=[]
        for line in Path(fold['path']).read_text().splitlines():
            match=re.search(r'folds:\s*(.*?)\s+object:\s*(\S+)',line)
            if match and match[1].split()[0]=='0':train_tools.append(match[2])
        chosen=next((t for t in sorted(train_tools) if any(x['archive']=='tools' and x['group']==t for x in frames.values())),None)
        if chosen:
            from scipy.io import whosmat
            examples=sorted((x for x in frames.values() if x['archive']=='tools' and x['group']==chosen and 'label.mat' in x),key=lambda x:x['frame_id'])[:3]
            for x in examples:
                p=x['label.mat']['path']
                structure.append({'tool':chosen,'frame_id':x['frame_id'],'native_fold1_membership':'train (first fold value 0)',
                                  'path':p,'variables':whosmat(p),'pixel_values_read':False})
    result={'status':'CENSUS_COMPLETE_MANUAL_INDEX_UNKNOWN','archive_entries':len(entries),
            'unique_frames':len(frames),'per_archive_unique_frames':dict(Counter(x['archive'] for x in frames.values())),
            'file_type_counts':dict(counts),'groups':{k:dict(v) for k,v in groups.items()},
            'rgb_label_pairs':sum('rgb.jpg' in x and 'label.mat' in x for x in frames.values()),
            'missing_rgb_or_label':[k for k,x in frames.items() if 'rgb.jpg' not in x or 'label.mat' not in x],
            'extracted_bytes':extracted_bytes,'excluded_from_extraction':['depth.png','label_rank.mat'],
            'native_class_names':NAMES,'umd_to_iit_native_mapping':IIT_MAP,'scoop_handling':'unmapped; not background',
            'manual_index':'UNKNOWN; author says every third but no residue assumed',
            'manual_mask_count':None,'auto_mask_count':None,'mask_structure_train_only':structure,
            'test_label_pixels_read':False,'RGB_pixels_decoded':False,'new_model_runs':0}
    save(out/'census.json',result)
    save(out/'terminal.json',{'status':result['status'],'census':str(out/'census.json'),'download':receipt,'resources_released':True})
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','missing_rgb_or_label','mask_structure_train_only']}),flush=True)


def resolve_index(out):
    """Label provenance classification from original author code; no masks read."""
    out=Path(out)
    source=out/'author_hmp_source/part_affordance_release/src'
    evidence_paths={'tools_selector':source/'config_files/config_compare_feats.m',
                    'clutter_selector':source/'config_files/config_clutter.m',
                    'token_parser':source/'config_files/config_base.m',
                    'filter_application':source/'util/filter_file_info.m'}
    evidence={}
    for name,path in evidence_paths.items():
        text=path.read_text(encoding='utf-8-sig')
        pattern='mod(str2num(x(end-7:end)), 3) == 1' if 'selector' in name else 'token_expr' if name=='token_parser' else 'cellfun(select_function'
        matches=[{'line':i,'text':line.strip()} for i,line in enumerate(text.splitlines(),1) if pattern in line]
        if not matches:raise ValueError(f'Author code evidence missing: {name}')
        evidence[name]={'path':str(path),'sha256':sha(path),'matching_lines':matches}
    receipt=json.loads((out/'code_request_reservations.json').read_text())
    provenance={'source_url':receipt['url'],'archive_sha256':receipt['sha256'],
                'official_project_page':'https://users.umiacs.umd.edu/~fermulcm/affordance/Affordance.html',
                'native_readme':str(BASE/'metadata/umd_README.txt'),
                'rule':'integer final 8 digits of RGB frame token modulo 3 equals 1',
                'basis':'Official README says every third is manually annotated and other frames excluded from experiments; original author training and clutter evaluation code selects remainder 1.',
                'claim_limit':'Manual provenance reconstructed from original code+README; no per-mask hand flag asserted and no mask pixel read',
                'files':evidence}
    save(out/'manual_index_provenance.json',provenance)
    manifest=out/'frame_manifest.jsonl'
    if not manifest.exists():
        print('Author provenance saved; full frame census pending',flush=True);return
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    counts=defaultdict(Counter);manual=[];unknown=[]
    for r in rows:
        match=re.search(r'_(\d{8})$',r['frame_id'])
        if not match:
            unknown.append(r['source_id']);continue
        index=int(match[1]);is_manual=index%3==1
        r['frame_number']=index;r['manual_annotation_status']='AUTHOR_SELECTED_MANUAL' if is_manual else 'AUTHOR_DESCRIBED_AUTOMATIC'
        r['manual_index_provenance']=str(out/'manual_index_provenance.json')
        counts[r['archive']]['manual_frames' if is_manual else 'automatic_frames']+=1
        counts[r['archive']][f'remainder_{index%3}']+=1
        if is_manual and 'rgb.jpg' in r and 'label.mat' in r:manual.append(r)
    with manifest.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    with (out/'manual_frame_manifest.jsonl').open('w',encoding='utf-8') as f:
        for r in manual:f.write(json.dumps(r)+'\n')
    summary={'native_unique_frames':len(rows),'manual_rgb_label_pairs':len(manual),
             'per_archive':{k:dict(v) for k,v in counts.items()},'unresolved_frame_ids':unknown,
             'pixel_values_read':False,'provenance':str(out/'manual_index_provenance.json')}
    save(out/'manual_index_census.json',summary)
    census_path=out/'census.json'
    census_data=json.loads(census_path.read_text())
    census_data.update(status='CENSUS_COMPLETE_AUTHOR_INDEX_RESOLVED' if not unknown else 'CENSUS_COMPLETE_AUTHOR_INDEX_PARTIAL',
        manual_index=provenance['rule'],manual_index_provenance=str(out/'manual_index_provenance.json'),
        manual_mask_count=sum(r['manual_annotation_status']=='AUTHOR_SELECTED_MANUAL' and 'label.mat' in r for r in rows),
        auto_mask_count=sum(r['manual_annotation_status']=='AUTHOR_DESCRIBED_AUTOMATIC' and 'label.mat' in r for r in rows))
    save(census_path,census_data)
    terminal=json.loads((out/'terminal.json').read_text())
    terminal['status']=census_data['status'];terminal['manual_index_census']=str(out/'manual_index_census.json')
    auxiliary=json.loads((out/'code_request_reservations.json').read_text())
    terminal['author_code_reserved_bytes']=auxiliary['reserved_bytes']
    terminal['total_new_request_reserved_upper_bound_bytes']=terminal['download']['new_request_reserved_upper_bound_bytes']+auxiliary['reserved_bytes']
    if terminal['total_new_request_reserved_upper_bound_bytes']>3600000000:raise ValueError('Budget exceeded')
    save(out/'terminal.json',terminal)
    print(json.dumps(summary),flush=True)


def shared_records(out):
    """Emit future evaluation inputs; never load a MAT array or image."""
    out=Path(out)
    paths=json.loads((out/'native_text_files.json').read_text())
    fold_path=next(Path(x['path']) for x in paths if Path(x['path']).name=='category_10_fold.txt')
    folds={}
    for line in fold_path.read_text().splitlines():
        m=re.search(r'folds:\s*(.*?)\s+object:\s*(\S+)',line)
        if m:folds[m[2]]=[int(x) for x in m[1].split()]
    manual=[json.loads(x) for x in (out/'manual_frame_manifest.jsonl').read_text().splitlines() if x.strip()]
    records=[]
    for r in manual:
        records.append({'source_id':r['source_id'],'dataset':'UMD','split':'cross_source_pending',
            'image':r['rgb.jpg']['path'],'mask':r['label.mat']['path'],'mask_format':'matlab','mask_variable':'gt_label',
            'native_class_ids':list(range(8)),'native_class_names':NAMES,'native_to_IIT':IIT_MAP,
            'unmapped_label_policy':'UMD scoop=3 must be ignore=255 for shared-label evaluation, never background',
            'supervision_scope':'full_image','source_group':f"umd/{r['archive']}/{r['group']}",
            'image_source':r['archive'],'native_tool_or_scene':r['group'],
            'native_ten_fold_test_membership':folds.get(r['group']),
            'native_split_note':'1 is held-out tool for that fold; no UMD labels used for training here',
            'manual_annotation_status':r['manual_annotation_status'],
            'shared_class_presence':'UNKNOWN: no new mask pixels decoded',
            'manual_index_provenance':r['manual_index_provenance']})
    with (out/'shared_label_records.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    info={'records':len(records),'unique_source_groups':len({r['source_group'] for r in records}),
          'native_shared_label_pairs':IIT_MAP,'status':'READY_FOR_AUTHORIZED_CROSS_SOURCE_EVALUATION_NO_PIXELS_READ',
          'scientific_scores_computed':False,'native_mask_pixels_read':False,
          'manual_index_census':str(out/'manual_index_census.json')}
    save(out/'shared_label_records_receipt.json',info)
    terminal=json.loads((out/'terminal.json').read_text());terminal['shared_label_records']=info
    save(out/'terminal.json',terminal);print(json.dumps(info),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['download','census','all','resolve-index','records'])
    p.add_argument('--output',default=str(BASE/'data/umd'))
    p.add_argument('--budget-bytes',type=int,default=3600000000)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--chunk-bytes',type=int,default=8*1024**2)
    a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('workers must be 1..8')
    if a.budget_bytes>3600000000:raise ValueError('Exceeds authorized total budget')
    if a.command in ['download','all']:Download(a.output,a.budget_bytes,a.workers,a.chunk_bytes).run()
    if a.command in ['census','all']:census(a.output)
    if a.command in ['resolve-index','all']:resolve_index(a.output)
    if a.command in ['records','all']:shared_records(a.output)


if __name__=='__main__':main()
