"""Controller-owned durable capture queue; workers never share its filesystem.

This module records scheduling only; it launches no jobs and establishes no
statistical independence. Existing jobs can be imported with register_running.
Never infer completion from process exit: finish requires an explicit outcome.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import json
import ntpath
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

SCHEMA='cnh-controller-capture-queue-v1'


def _nonempty(value, name):
    if not isinstance(value,str) or not value.strip():
        raise ValueError(name+' must be a nonempty string')
    return value


class CaptureQueue:
    def __init__(self, path):
        self.path=Path(path)

    @contextmanager
    def _transaction(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        lock=self.path.with_name(self.path.name+'.lock')
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as exc:
            raise RuntimeError('Queue locked; no automatic stale-lock stealing: '+str(lock)) from exc
        temporary=None
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(dict(pid=os.getpid(),created_unix=time.time()),stream)
                stream.flush();os.fsync(stream.fileno())
            state=json.loads(self.path.read_text()) if self.path.exists() else dict(
                schema=SCHEMA,revision=0,tasks={},dispatch_history=[],
                independence='NOT_ESTABLISHED_BY_SCHEDULING_OR_MACHINE_ROTATION')
            if state.get('schema')!=SCHEMA:
                raise ValueError('Unsupported queue schema')
            yield state
            state['revision']+=1
            fd,temporary=tempfile.mkstemp(prefix=self.path.name+'.',suffix='.partial',dir=self.path.parent)
            with os.fdopen(fd,'w',encoding='utf-8') as stream:
                json.dump(state,stream,indent=2,allow_nan=False)
                stream.write('\n');stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,self.path)
            temporary=None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
            lock.unlink()

    def snapshot(self):
        # Atomic replacement makes an unlocked read coherent.
        return json.loads(self.path.read_text())

    def next_task(self, *, machine_id):
        """Advisory selection for resolving host paths; claim rechecks atomically."""
        _nonempty(machine_id,'machine_id')
        task=self._choose(self.snapshot(),machine_id)
        return copy.deepcopy(task) if task else None

    def add(self, task_id, *, configuration_sha256, category, layouts):
        _nonempty(task_id,'task_id');_nonempty(category,'category')
        if not isinstance(configuration_sha256,str) or not re.fullmatch('[0-9a-f]{64}',configuration_sha256):
            raise ValueError('Frozen logical configuration SHA256 required')
        if not layouts or len(set(layouts))!=len(layouts) or any(not isinstance(x,str) or not x for x in layouts):
            raise ValueError('Unique nonempty layout identities required')
        with self._transaction() as state:
            if task_id in state['tasks']:
                raise ValueError('Task already registered')
            state['tasks'][task_id]=dict(id=task_id,configuration_sha256=configuration_sha256,
                category=category,layouts=list(layouts),status='pending',attempts=[],checkpoints={},
                mechanical_failures=0,quality_quarantined_layouts=[],events=[])

    @staticmethod
    def _choose(state,machine_id):
        pending=[t for t in state['tasks'].values() if t['status']=='pending']
        if not pending:
            return None
        # Rotate categories per machine, then balance global category dispatches.
        history=state['dispatch_history']
        machine_history=[h for h in history if h['machine_id']==machine_id]
        last=machine_history[-1]['category'] if machine_history else None
        def score(task):
            category=task['category']
            return (category==last,sum(h['category']==category for h in history),
                    sum(h['category']==category for h in machine_history),task['id'])
        return min(pending,key=score)

    @staticmethod
    def _start(state,task,machine_id,spec_path,output_path,configuration_sha256,origin):
        for value,name in [(machine_id,'machine_id'),(spec_path,'spec_path'),(output_path,'output_path')]:
            _nonempty(value,name)
        # Compare lexically normalized Windows/Unix paths across all attempts.
        if configuration_sha256!=task['configuration_sha256']:
            raise ValueError('Selected task and supplied frozen logical configuration differ')
        if any(t['status']=='running' and t['attempts'][-1]['machine_id']==machine_id for t in state['tasks'].values()):
            raise ValueError('Machine already owns a running task')
        normalized=lambda p:ntpath.normpath(p).replace('\\','/').rstrip('/').casefold()
        if any(normalized(a['output_path'])==normalized(output_path)
               for t in state['tasks'].values() for a in t['attempts']):
            raise ValueError('Output path already used; failed evidence must never be overwritten')
        token=uuid.uuid4().hex
        attempt=dict(number=len(task['attempts'])+1,token=token,machine_id=machine_id,
            spec_path=spec_path,output_path=output_path,configuration_sha256=task['configuration_sha256'],
            origin=origin,started_unix=time.time(),status='running',
            remaining_layouts=[x for x in task['layouts'] if x not in task['checkpoints']])
        task['attempts'].append(attempt);task['status']='running'
        state['dispatch_history'].append(dict(task_id=task['id'],machine_id=machine_id,
            category=task['category'],attempt=attempt['number'],origin=origin))
        return dict(task_id=task['id'],**copy.deepcopy(attempt))

    def claim(self, *, machine_id,spec_path,output_path,configuration_sha256):
        with self._transaction() as state:
            if any(t['status']=='running' and t['attempts'][-1]['machine_id']==machine_id for t in state['tasks'].values()):
                raise ValueError('Machine already owns a running task; resume or finish it explicitly')
            task=self._choose(state,machine_id)
            return self._start(state,task,machine_id,spec_path,output_path,configuration_sha256,'QUEUE_CLAIM') if task else None

    def register_running(self,task_id,*,machine_id,spec_path,output_path,configuration_sha256,evidence):
        """Import an already dispatched job honestly; no claim is fabricated."""
        _nonempty(evidence,'evidence')
        with self._transaction() as state:
            task=state['tasks'][task_id]
            if task['status']!='pending' or task['attempts']:
                raise ValueError('Existing-job registration requires an unclaimed pending task')
            result=self._start(state,task,machine_id,spec_path,output_path,configuration_sha256,'OBSERVED_EXISTING_JOB')
            task['attempts'][-1]['registration_evidence']=evidence
            return result

    @staticmethod
    def _active(state,task_id,token):
        task=state['tasks'][task_id]
        if task['status']!='running' or not task['attempts'] or task['attempts'][-1]['token']!=token:
            raise ValueError('Stale token or task is not running')
        return task,task['attempts'][-1]

    def checkpoint(self,task_id,token,*,layout_id,outcome,evidence):
        if outcome not in ('completed','quarantined'):
            raise ValueError('Layout outcome must be completed or quarantined')
        _nonempty(evidence,'evidence')
        with self._transaction() as state:
            task,attempt=self._active(state,task_id,token)
            if layout_id not in task['layouts']:
                raise ValueError('Layout outside frozen task')
            record=dict(outcome=outcome,evidence=evidence,attempt=attempt['number'])
            if layout_id in task['checkpoints']:
                if task['checkpoints'][layout_id]!=record:
                    raise ValueError('Checkpoint is immutable; cannot turn quality failure into success')
                return
            task['checkpoints'][layout_id]=record
            if outcome=='quarantined':
                task['quality_quarantined_layouts'].append(layout_id)

    def finish(self,task_id,token,*,outcome,evidence):
        if outcome not in ('completed','mechanical_failed'):
            raise ValueError('Quality failures use layout quarantine checkpoints, not retries')
        _nonempty(evidence,'evidence')
        with self._transaction() as state:
            task,attempt=self._active(state,task_id,token)
            if outcome=='completed' and set(task['checkpoints'])!=set(task['layouts']):
                raise ValueError('Every layout needs an explicit terminal checkpoint')
            attempt.update(status=outcome,finished_unix=time.time(),evidence=evidence)
            if outcome=='mechanical_failed':
                task['mechanical_failures']+=1
                task['status']='mechanical_failed'
            else:
                task['status']='quarantined' if task['quality_quarantined_layouts'] else 'completed'
                attempt['status']=task['status']

    def retry_mechanical(self,task_id,*,evidence):
        """Authorize the single retry of remaining layouts, with a future new path."""
        _nonempty(evidence,'evidence')
        with self._transaction() as state:
            task=state['tasks'][task_id]
            if task['status']!='mechanical_failed' or task['mechanical_failures']!=1 or len(task['attempts'])!=1:
                raise ValueError('Only one mechanical retry is permitted; quality failure cannot retry')
            if len(task['checkpoints'])==len(task['layouts']):
                raise ValueError('All layouts terminal; retry would repeat completed evidence')
            task['status']='pending'
            task['events'].append(dict(event='ONE_MECHANICAL_RETRY_AUTHORIZED',evidence=evidence))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--request',type=Path,help='JSON object with operation and keyword arguments; omitted reads snapshot')
    args=parser.parse_args();queue=CaptureQueue(args.queue)
    if args.request:
        request=json.loads(args.request.read_text());operation=request.pop('operation')
        if operation not in ('add','next_task','claim','register_running','checkpoint','finish','retry_mechanical'):
            raise ValueError('Unsupported operation')
        result=getattr(queue,operation)(**request)
    else:
        result=queue.snapshot()
    print(json.dumps(result,indent=2))
