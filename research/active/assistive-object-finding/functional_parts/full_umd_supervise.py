"""Adopt one existing task-owned UMD downloader, then finish metadata-only steps.

Never creates a second writer while the adopted process is alive. A stop file
terminates only matching owned PIDs (creation time and command checked).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from full_umd_data import BASE, ROOT, save


def identity(pid):
    p=psutil.Process(pid)
    command=p.cmdline()
    if not any('full_umd_data.py' in x for x in command):raise ValueError('PID is not task UMD downloader')
    return {'pid':pid,'create_time':p.create_time(),'command':command}


def alive(owner):
    try:
        p=psutil.Process(owner['pid'])
        return abs(p.create_time()-owner['create_time'])<.01 and p.is_running()
    except psutil.NoSuchProcess:return False


def stop_owned(owners):
    for owner in owners:
        if alive(owner):
            p=psutil.Process(owner['pid'])
            if any('full_umd_data.py' in x for x in p.cmdline()):
                # Windows venv launches a second Python interpreter. Terminate
                # only matching descendants before their known launcher.
                for child in reversed(p.children(recursive=True)):
                    try:
                        if any('full_umd_data.py' in x for x in child.cmdline()):child.terminate()
                    except psutil.NoSuchProcess:pass
                try:p.terminate()
                except psutil.NoSuchProcess:pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-pids',nargs='*',type=int,default=[])
    parser.add_argument('--output',default=str(BASE/'data/umd'))
    args=parser.parse_args();out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    owner_path=out/'supervisor_owner.json';stop_file=out/'STOP.flag'
    previous={}
    if owner_path.exists():
        previous=json.loads(owner_path.read_text())
        old=previous.get('supervisor',{})
        if old and alive(old):raise RuntimeError('Existing supervisor is alive')
    supervisor={'pid':os.getpid(),'create_time':psutil.Process().create_time()}
    if args.existing_pids:
        owners=[identity(pid) for pid in args.existing_pids]
    else:
        # A crashed supervisor may leave a healthy downloader. Re-adopt it from
        # verified ownership rather than start a second archive writer.
        candidates=previous.get('adopted_downloaders',[])
        if previous.get('active_owned_child'):candidates=candidates+[previous['active_owned_child']]
        owners=[identity(x['pid']) for x in candidates if alive(x)]
    state={'supervisor':supervisor,'adopted_downloaders':owners,'existing_tool_session_id':84486,
           'stop_file':str(stop_file),'status':'WAITING_ADOPTED_DOWNLOAD','metadata_only':True,
           'resume_command':[sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(out)]}
    save(owner_path,state)
    script=Path(__file__).with_name('full_umd_data.py')
    def event(stage,**kwargs):
        with (out/'supervisor_progress.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'time':time.time(),'stage':stage,**kwargs})+'\n')
    def check_stop():
        if stop_file.exists():
            stop_owned(owners)
            state['status']='STOPPED_BY_FILE';save(owner_path,state)
            save(out/'supervisor_terminal.json',{'status':'STOPPED_BY_FILE','owned_workers_terminated':True,'partial_preserved':True})
            raise SystemExit(0)
    def run(command,chunk=None):
        cmd=[sys.executable,'-B',str(script),command,'--output',str(out)]
        if chunk:cmd.extend(['--chunk-bytes',str(chunk)])
        log=out/f'supervisor_{command.replace("-","_")}.log'
        with log.open('a',encoding='utf-8') as handle:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            owners[:]=[identity(child.pid)];state['active_owned_child']=owners[0];state['status']='RUNNING_'+command
            save(owner_path,state);event('command_start',command=command,pid=child.pid)
            while child.poll() is None:check_stop();time.sleep(15)
            event('command_end',command=command,returncode=child.returncode)
            if child.returncode:raise RuntimeError(f'{command} failed; see {log}')
    try:
        event('adopted',pids=[x['pid'] for x in owners])
        while any(alive(x) for x in owners):check_stop();time.sleep(15)
        check_stop()
        # If the original process fails, one smaller-range recovery is allowed
        # within the unchanged persistent request budget. No infinite retries.
        if not (out/'download_receipt.json').exists():run('download',512*1024)
        if not (out/'census.json').exists():run('census')
        run('resolve-index');run('records')
        final=json.loads((out/'terminal.json').read_text())
        result={'status':'COMPLETE','terminal':str(out/'terminal.json'),
                'manual_index':str(out/'manual_index_census.json'),'records':str(out/'shared_label_records.jsonl'),
                'scientific_scores_computed':False,'new_mask_pixels_read':False,
                'resources_released':True,'data_terminal_status':final['status']}
        save(out/'supervisor_terminal.json',result);state['status']='COMPLETE';save(owner_path,state)
    except Exception as exc:
        stop_owned(owners)
        state['status']='INCOMPLETE';save(owner_path,state)
        save(out/'supervisor_terminal.json',{'status':'INCOMPLETE','error':f'{type(exc).__name__}: {exc}',
             'partial_preserved':True,'resources_released':True,'resume_command':state['resume_command']})
        raise


if __name__=='__main__':main()
