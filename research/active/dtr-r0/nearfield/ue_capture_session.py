"""Bounded editor session: isolate each capture's globals and exit after the queue."""
import json
import os
from pathlib import Path
import time
import traceback
import unreal as u

ROOT = Path(os.environ['BA_UE_SESSION'])
REQUEST = json.loads((ROOT/'request.json').read_text(encoding='utf-8'))
jobs = iter(REQUEST['jobs'])
state = dict(map_loaded=None)
current = None
namespace = None
pending = True
finished = False
started = time.monotonic()
rows = []


def finish(error=None):
    global finished
    if finished:
        return
    finished = True
    (ROOT/'receipt.json').write_text(json.dumps(dict(status='FAIL' if error else 'PASS',
        error=error, pid=os.getpid(), blocks=rows, wall_elapsed_s=time.monotonic()-started), indent=2), encoding='utf-8')
    u.unregister_slate_post_tick_callback(handle)
    u.SystemLibrary.quit_editor()


def complete(report):
    global pending
    rows.append(dict(id=current['id'], output=current['output'], status=report['status']))
    if report['status'] != 'PASS':
        finish('Block failed: ' + current['id'])
    else:
        pending = True


state['on_complete'] = complete


def tick(delta):
    global pending, current, namespace
    if finished:
        return
    if (ROOT/'stop.request').exists():
        if namespace and callable(namespace.get('finish')):
            namespace['finish']('Batch owner requested stop')
        finish('Batch owner requested stop')
        return
    if not pending:
        return
    pending = False  # exec/load can reenter Slate.
    try:
        current = next(jobs, None)
        if current is None:
            finish()
            return
        api = u.BlindAssistCaptureLibrary
        if not hasattr(api, 'reset_capture_pairs') or not hasattr(api, 'reset_rgb_writes'):
            raise RuntimeError('Batch sessions require the reset-enabled capture plugin')
        os.environ.update(current['env'])
        script = Path(current['script'])
        namespace = dict(__name__='__capture_block__', __file__=str(script), __ba_session__=state)
        exec(compile(script.read_text(encoding='utf-8'), str(script), 'exec'), namespace)
    except Exception:
        finish(traceback.format_exc())


handle = u.register_slate_post_tick_callback(tick)
