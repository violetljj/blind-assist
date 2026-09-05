"""Measure the live visual assets in an owned, unsaved editor world."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import unreal as u

sys.path.insert(0,str(Path(__file__).parent))
import street_scenarios as scenarios
import visual_geometry
OUT=Path(os.environ['BA_UE_GEOMETRY_OUTPUT']).resolve()
if not OUT.is_relative_to((Path(__file__).resolve().parents[4]/'artifacts.local').resolve()):
    raise ValueError('Geometry measurement output must remain under artifacts.local')
OUT.mkdir(parents=True,exist_ok=False)
api=u.get_editor_subsystem(u.EditorActorSubsystem)
world=u.EditorLoadingAndSavingUtils.new_blank_map(False)
human_assets=json.loads((Path(u.Paths.project_dir())/'Saved/lab-visual-upgrade.json').read_text())['humans']
source=Path(__file__).with_name('capture_street_closed_loop.py')
functions=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('spawn_actor','move_visuals')]
spawned=[];ANCHOR_X=0.;ANCHOR_Y=0.;floor=0.;last_yaw={}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),globals())
spec=scenarios.scenario_catalog()[0]
visuals={s['id']:spawn_actor(s) for s in spec['actors']}
report={'status':'RUNNING','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'engine_version':u.SystemLibrary.get_engine_version(),'asset':human_assets[0], 'samples':[]}
state={'i':0,'after':time.monotonic()+3}
def finish(error=None):
    report['status']='FAIL' if error else 'PASS'
    if error:report['error']=error
    (OUT/'result.json').write_text(json.dumps(report,indent=2))
    u.unregister_slate_post_tick_callback(handle)
    u.SystemLibrary.quit_editor()
def tick(delta):
    try:
        if time.monotonic()<state['after']:return
        t=state['i']*.05
        move_visuals(t)
        rows=[]
        for s in scenarios.actors_at(spec,t):
            a=visuals[s['id']];center,extent=visual_geometry.native_bounds(a)
            rows.append({'id':s['id'],'kind':s['kind'],'center_m':center,
                         'extent_m':extent, 'pose':s,
                         'assessment':visual_geometry.assess_bounds(s,center,extent,[s['x_m'],s['y_m']])})
        report['samples'].append({'t':t,'actors':rows})
        state['i']+=1
        if state['i']>=25:finish();return
        state['after']=time.monotonic()+.02
    except Exception:finish(traceback.format_exc())
handle=u.register_slate_post_tick_callback(tick)
