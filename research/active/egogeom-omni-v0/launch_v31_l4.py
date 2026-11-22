"""Detached, websocket-drop-proof launcher for EgoGeom-Omni v3.1 Pinnacle Polish on L4/G4."""
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

PID_FILE = Path("/content/train_v31.pid")
LOG_FILE = Path("/content/train_v31.log")
EXIT_FILE = Path("/content/train_v31.exit")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(
    f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'} | "
    f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB | "
    f"CPUs: {os.cpu_count()} | Disk /content free: {shutil.disk_usage('/content').free/1e9:.1f} GB | "
    f"Drive mounted: {os.path.exists('/content/drive/MyDrive/BlindAssist/datasets')}",
    flush=True,
)

pid = None
if PID_FILE.exists() and not EXIT_FILE.exists():
    try:
        cand = int(PID_FILE.read_text().strip())
        if _pid_alive(cand):
            pid = cand
            print(f"[Launcher v3.1] Re-attaching to already-running trainer PID {pid}!", flush=True)
    except Exception:
        pass

if pid is None:
    EXIT_FILE.unlink(missing_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    runner_cmd = (
        f"{sys.executable} -u /content/train_egogeom_omni_v31_polish.py "
        f">> {LOG_FILE} 2>&1; echo $? > {EXIT_FILE}"
    )
    LOG_FILE.write_text("", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", "-c", runner_cmd],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    pid = proc.pid
    PID_FILE.write_text(str(pid), encoding="utf-8")
    print(f"[Launcher v3.1] Started detached trainer PID {pid} -> {LOG_FILE}", flush=True)

pos = 0
last_out = time.time()
while True:
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            if chunk:
                print(chunk, end="", flush=True)
                pos = f.tell()
                last_out = time.time()
    if EXIT_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            rem = f.read()
            if rem:
                print(rem, end="", flush=True)
        ret = int(EXIT_FILE.read_text().strip() or "1")
        print(f"=== Detached v3.1 trainer exited with code {ret} ===", flush=True)
        if ret == 0:
            os.sync()
            out_dir = Path("/content/drive/MyDrive/BlindAssist/runs/egogeom_omni_v31")
            if out_dir.exists():
                for fp in sorted(out_dir.iterdir()):
                    print(f"  [Drive] {fp.name}: {fp.stat().st_size/1e6:.2f} MB", flush=True)
        else:
            raise SystemExit(ret)
        break
    if not _pid_alive(pid):
        time.sleep(2.0)
        if not EXIT_FILE.exists():
            print(f"=== Trainer PID {pid} ended without exit file ===", flush=True)
            break
    if time.time() - last_out > 45.0:
        print(f"  [heartbeat {time.strftime('%H:%M:%S')}] v3.1 trainer PID {pid} alive...", flush=True)
        last_out = time.time()
    time.sleep(2.0)
