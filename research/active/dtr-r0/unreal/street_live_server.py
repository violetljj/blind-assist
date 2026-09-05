"""Local lockstep DTR sensor service; model-root is the only input filesystem tree."""
import argparse
import json
import os
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np
import ue_dtr_replay as replay
from street_live_policy import MotionPolicy,depth_corridors
from ue_replay_cache import cached_replay_inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(args.output/'yolo-config')
    import torch
    from ultralytics import YOLO
    weights=replay.detector.DEFAULT_MODEL
    model=YOLO(str(weights),task='segment')
    device='0' if torch.cuda.is_available() else 'cpu'
    model.to(device='cuda:0' if device=='0' else 'cpu')
    names=replay.detector.model_names(model)
    weight_hash=replay.adapter.sha256_file(weights)
    histories={}
    policies={}
    response_cache={}
    backend={'torch':torch.__version__,'device':str(next(model.model.parameters()).device),
             'gpu':torch.cuda.get_device_name(0) if device=='0' else None,'model_sha256':weight_hash}
    (args.output/'backend.json').write_text(json.dumps(backend,indent=2))
    prior=args.output/'responses.jsonl'
    if prior.exists():
        for line in prior.read_text().splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue  # interrupted diagnostic journal; checkpoint is authoritative
            p=row['prediction']
            response_cache[(p['episode_id'],p['sample_index'])]=row
        for checkpoint in args.output.glob('*-checkpoint.json'):
            value=json.loads(checkpoint.read_text())
            if value['model_sha256']!=weight_hash: raise RuntimeError('Cannot resume with changed weights')
            policy=MotionPolicy()
            policy.__dict__.update(value['policy'])
            policies[checkpoint.name.removesuffix('-checkpoint.json')]=policy
            if 'last_response' in value:
                row=value['last_response']
                p=row['prediction']
                response_cache[(p['episode_id'],p['sample_index'])]=row

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*unused): pass
        def do_POST(self):
            started=time.perf_counter()
            try:
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if set(request)!={'episode_id','sample_index','goal_forward_m'}:
                    raise ValueError('Only episode/frame identity and public navigation goal accepted')
                key=(request['episode_id'],request['sample_index'])
                if key in response_cache:
                    result=response_cache[key]
                else:
                    contract=replay.load_contract(args.model_root)
                    episode=contract.episode(request['episode_id'])
                    observation=episode.observations[-1]
                    if observation.sample_index!=request['sample_index']:
                        raise ValueError('Request must name latest observed prefix frame')
                    candidates=histories.setdefault(episode.episode_id,[])
                    # Reconstruct an interrupted prefix using sensor frames only.
                    while len(candidates)<len(episode.observations):
                        o=episode.observations[len(candidates)]
                        source={'episode_id':o.episode_id,'sample_index':o.sample_index,'time_s':o.time_s,
                                'world_frame':o.world_frame,'frame_id':f'{o.episode_id}/{o.sample_index:06d}'}
                        c=replay.detector.infer_one(model,names,replay.detector.ImageInput(o.rgb.path,source,o.rgb.sha256),
                             model_path=weights,model_sha256=weight_hash,device=device,run_kind='LIVE_LOCKSTEP_SENSOR')
                        candidates.append(c)
                    if len(episode.observations)<2:
                        row={'episode_id':episode.episode_id,'sample_index':observation.sample_index,
                             'time_s':observation.time_s,'route_risk':False,'event':'WARMUP',
                             'risk_state':'UNKNOWN_INSUFFICIENT_HISTORY','support_state':'ONE_FRAME_ONLY',
                             'global_observability':'UNKNOWN_NOT_ESTIMATED_BY_X73'}
                    else:
                        with cached_replay_inputs():
                            prediction=replay.predict_episode(episode,candidates,contract.calibration)
                        row=replay.compact_rows(episode.episode_id,prediction)[-1]
                    depth=replay.load_linear_depth(observation,contract.calibration)
                    anchor=episode.route_frame.center_xy_m
                    transform=observation.wearer['transform']
                    x,y=transform['x']-anchor[0],transform['y']-anchor[1]
                    corridors=depth_corridors(depth,contract.calibration.horizontal_fov_degrees,
                        observation.camera_transform['z']-transform['z'],y,observation.camera_transform['pitch'])
                    policy=policies.setdefault(episode.episode_id,MotionPolicy())
                    command=policy.command(t=observation.time_s,x=x,y=y,goal_x=request['goal_forward_m'],
                         dtr_risk=row['route_risk'],corridors=corridors)
                    result={'prediction':row,'corridors':corridors,'command':command,
                            'elapsed_s':time.perf_counter()-started,'input_prefix_frames':len(candidates)}
                    response_cache[key]=result
                    with (args.output/'responses.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
                    # Per-step durable state is sufficient for exact controller resume.
                    checkpoint=args.output/(episode.episode_id+'-checkpoint.json')
                    temporary=checkpoint.with_suffix('.tmp')
                    temporary.write_text(json.dumps({
                        'last_sample_index':observation.sample_index,'policy':policy.__dict__,
                        'model_sha256':weight_hash,'manifest_sha256':contract.manifest_sha256,'last_response':result}))
                    temporary.replace(checkpoint)
                payload=json.dumps(result,allow_nan=False).encode()
                self.send_response(200)
            except Exception:
                import traceback
                payload=json.dumps({'error':traceback.format_exc()}).encode()
                self.send_response(500)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server=HTTPServer(('127.0.0.1',args.port),Handler)
    (args.output/'ready.json').write_text(json.dumps({'port':server.server_port,**backend}))
    print('LIVE_DTR_READY',server.server_port,flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__=='__main__': main()
