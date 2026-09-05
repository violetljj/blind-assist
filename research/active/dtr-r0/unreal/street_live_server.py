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


def restore_checkpoints(output, controller_mode, prediction_engine, weight_hash, action_footprint_state='frozen'):
    """Only an atomically committed checkpoint may advance resumed policy state."""
    policies, responses, committed = {}, {}, {}
    for checkpoint in output.glob('*-checkpoint.json'):
        value=json.loads(checkpoint.read_text())
        if value['model_sha256']!=weight_hash:
            raise RuntimeError('Cannot resume with changed weights')
        if value.get('controller_mode','JOINT')!=controller_mode:
            raise RuntimeError('Cannot resume a different controller mode')
        if value.get('prediction_engine','batch')!=prediction_engine:
            raise RuntimeError('Cannot resume a different prediction engine')
        if value.get('action_footprint_state','frozen')!=action_footprint_state:
            raise RuntimeError('Cannot resume a different action footprint state')
        episode_id=checkpoint.name.removesuffix('-checkpoint.json')
        policy=MotionPolicy(controller_mode)
        policy.__dict__.update(value['policy'])
        policies[episode_id]=policy
        committed[episode_id]=value['last_sample_index']
        if 'last_response' in value:
            row=value['last_response']
            prediction=row['prediction']
            if (prediction['episode_id'],prediction['sample_index'])!=(episode_id,value['last_sample_index']):
                raise RuntimeError('Checkpoint response and committed frame differ')
            responses[(episode_id,value['last_sample_index'])]=row
    journal=output/'responses.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            prediction=row['prediction']
            key=(prediction['episode_id'],prediction['sample_index'])
            # A crash between journal append and checkpoint commit must replay
            # this step from cached detections instead of skipping policy state.
            if key[0] in committed and key[1]<=committed[key[0]]:
                responses.setdefault(key,row)
    return policies,responses


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--controller-mode',choices=MotionPolicy.MODES,default='DEPTH_ONLY',
                        help='Default: DEPTH_ONLY, the current measured UE reference')
    parser.add_argument('--prediction-engine',choices=('incremental','batch'),default='incremental')
    parser.add_argument('--action-footprint-state',choices=('cadence','frozen'),default='cadence')
    args=parser.parse_args()
    if args.controller_mode.startswith('CANDIDATE_') and args.prediction_engine!='incremental':
        parser.error('Candidate motion queries require the shared incremental footprint state')
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
    engines={}
    latest_rows={}
    action_engines={}
    action_frames={}
    policies,response_cache=restore_checkpoints(args.output,args.controller_mode,args.prediction_engine,weight_hash,args.action_footprint_state)
    backend={'controller_mode':args.controller_mode,'prediction_engine':args.prediction_engine,
             'action_footprint_state':args.action_footprint_state,
             'torch':torch.__version__,'device':str(next(model.model.parameters()).device),
             'gpu':torch.cuda.get_device_name(0) if device=='0' else None,'model_sha256':weight_hash}
    (args.output/'backend.json').write_text(json.dumps(backend,indent=2))
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
                    timings={'detector':0.0,'prediction':0.0}
                    phase=time.perf_counter()
                    contract=replay.load_contract(args.model_root)
                    timings['contract']=time.perf_counter()-phase
                    episode=contract.episode(request['episode_id'])
                    observation=episode.observations[-1]
                    if observation.sample_index!=request['sample_index']:
                        raise ValueError('Request must name latest observed prefix frame')
                    if args.prediction_engine=='incremental' and episode.episode_id not in engines:
                        phase=time.perf_counter()
                        engines[episode.episode_id]=replay.IncrementalX73(
                            episode.episode_id,episode.route_frame,contract.calibration)
                        timings['engine_init']=time.perf_counter()-phase
                    engine=engines.get(episode.episode_id)
                    if args.controller_mode.startswith('CANDIDATE_') and args.action_footprint_state=='cadence' and episode.episode_id not in action_engines:
                        from ue_action_footprints import ActionFootprints
                        action_engines[episode.episode_id]=ActionFootprints(episode.episode_id,episode.route_frame,contract.calibration)
                    candidates=histories.setdefault(episode.episode_id,[]) if engine is None else None
                    count=engine.update_count if engine is not None else len(candidates)
                    # Reconstruct an interrupted prefix using sensor frames only.
                    while count<len(episode.observations):
                        o=episode.observations[count]
                        source={'episode_id':o.episode_id,'sample_index':o.sample_index,'time_s':o.time_s,
                                'world_frame':o.world_frame,'frame_id':f'{o.episode_id}/{o.sample_index:06d}'}
                        cache=args.output/'candidate-cache'/episode.episode_id/f'{o.sample_index:06d}.json'
                        cache_identity={'model_sha256':weight_hash,'rgb_sha256':o.rgb.sha256,'source':source}
                        if cache.exists():
                            cached=json.loads(cache.read_text())
                            if cached['identity']!=cache_identity:
                                raise RuntimeError('Candidate cache no longer matches the source/model')
                            c=cached['candidate']
                        else:
                            phase=time.perf_counter()
                            c=replay.detector.infer_one(model,names,replay.detector.ImageInput(o.rgb.path,source,o.rgb.sha256),
                                 model_path=weights,model_sha256=weight_hash,device=device,run_kind='LIVE_LOCKSTEP_SENSOR')
                            timings['detector']+=time.perf_counter()-phase
                            cache.parent.mkdir(parents=True,exist_ok=True)
                            temporary=cache.with_suffix('.tmp')
                            temporary.write_text(json.dumps({'identity':cache_identity,'candidate':c},allow_nan=False))
                            temporary.replace(cache)
                        if engine is not None:
                            phase=time.perf_counter()
                            latest_rows[episode.episode_id]=engine.update(o,c)
                            timings['prediction']+=time.perf_counter()-phase
                        else:
                            candidates.append(c)
                        if episode.episode_id in action_engines:
                            phase=time.perf_counter()
                            action_frames[episode.episode_id]=action_engines[episode.episode_id].update(o,c)
                            timings['action_footprints']=timings.get('action_footprints',0.0)+time.perf_counter()-phase
                        count+=1
                    if engine is not None:
                        row=latest_rows[episode.episode_id]
                    elif len(episode.observations)<2:
                        row={'episode_id':episode.episode_id,'sample_index':observation.sample_index,
                             'time_s':observation.time_s,'route_risk':False,'event':'WARMUP',
                             'risk_state':'UNKNOWN_INSUFFICIENT_HISTORY','support_state':'ONE_FRAME_ONLY',
                             'global_observability':'UNKNOWN_NOT_ESTIMATED_BY_X73'}
                    else:
                        phase=time.perf_counter()
                        with cached_replay_inputs():
                            prediction=replay.predict_episode(episode,candidates,contract.calibration)
                        row=replay.compact_rows(episode.episode_id,prediction)[-1]
                        timings['prediction']+=time.perf_counter()-phase
                    phase=time.perf_counter()
                    depth=replay.load_linear_depth(observation,contract.calibration)
                    transform=observation.wearer['transform']
                    (x,y),_=replay.x24.wearer_anchor_state(observation,episode.route_frame)
                    corridors=depth_corridors(depth,contract.calibration.horizontal_fov_degrees,
                        observation.camera_transform['z']-transform['z'],y,observation.camera_transform['pitch'])
                    policy=policies.setdefault(episode.episode_id,MotionPolicy(args.controller_mode))
                    motion_frame=action_frames.get(episode.episode_id,engine.last_rigid_frame if engine is not None else None)
                    command=policy.command(t=observation.time_s,x=x,y=y,goal_x=request['goal_forward_m'],
                         dtr_risk=row['route_risk'],corridors=corridors,
                         motion_frame=motion_frame)
                    timings['depth_and_action']=time.perf_counter()-phase
                    result={'prediction':row,'corridors':corridors,'command':command,
                            'elapsed_s':time.perf_counter()-started,'input_prefix_frames':count,
                            'timings_s':timings,'prediction_engine':args.prediction_engine,
                            'action_footprint_state':args.action_footprint_state,
                            'action_footprint_contract':motion_frame.get('fit_contract') if motion_frame else None,
                            'incremental_stats':engine.stats if engine is not None else None}
                    with (args.output/'responses.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
                    # Per-step durable state is sufficient for exact controller resume.
                    checkpoint=args.output/(episode.episode_id+'-checkpoint.json')
                    temporary=checkpoint.with_suffix('.tmp')
                    temporary.write_text(json.dumps({
                        'controller_mode':args.controller_mode,
                        'prediction_engine':args.prediction_engine,
                        'action_footprint_state':args.action_footprint_state,
                        'last_sample_index':observation.sample_index,'policy':policy.__dict__,
                        'model_sha256':weight_hash,'manifest_sha256':contract.manifest_sha256,'last_response':result}))
                    temporary.replace(checkpoint)
                    response_cache[key]=result
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
