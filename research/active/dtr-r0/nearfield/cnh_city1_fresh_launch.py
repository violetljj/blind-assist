"""Launch the existing City source collector with task-local pinned OpenEXR."""
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEPS = REPO/'artifacts.local/work/cnh-route-comparison-20260924/runtime/city1-python-deps'
if not (DEPS/'OpenEXR.cp311-win_amd64.pyd').is_file():
    raise RuntimeError('Pinned task-local OpenEXR==3.4.15 is missing')
sys.path.insert(0, str(DEPS))
runpy.run_path(str(HERE/'cnh_route_source_launch.py'), run_name='__main__')
