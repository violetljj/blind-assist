"""Single task-owned worker job: native cache, matched fits, durable receipt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('capture','cache','pretrained','initial','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=Path(__file__).resolve().parent;start=time.perf_counter()
    commands=[([sys.executable,'-B',str(source/'body_query_data.py'),'--capture',str(a.capture),'--output',str(a.cache)],'cache'),
              ([sys.executable,'-B',str(source/'body_query_train.py'),'--cache',str(a.cache),'--pretrained',str(a.pretrained),
                '--initial',str(a.initial),'--output',str(a.output)],'train')]
    completed=[]
    try:
        for cmd,name in commands:
            subprocess.run(cmd,check=True,cwd=source)
            completed.append(name)
    finally:
        path=a.output.parent/'model-process-release.json'
        path.write_text(json.dumps(dict(completed_stages=completed,subprocesses_joined=True,
            seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':main()
