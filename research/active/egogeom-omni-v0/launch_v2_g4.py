"""Detached, websocket-drop-proof launcher for EgoGeom-Omni v2 Stage 1 on G4.

Starts train_egogeom_omni_stage1_g4.py in its own POSIX session (`start_new_session=True`)
writing to /content/train_v2.log so local Colab CLI websocket drops NEVER interrupt G4 training!
If a trainer PID is already alive, re-attaches and tails its log instead of spawning a duplicate.
"""
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

PID_FILE = Path("/content/train_v21.pid")
LOG_FILE = Path("/content/train_v21.log")
EXIT_FILE = Path("/content/train_v21.exit")


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
            print(f"[Launcher] Re-attaching to already-running trainer PID {pid}!", flush=True)
    except Exception:
        pass

if pid is None:
    EXIT_FILE.unlink(missing_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    runner_cmd = (
        f"{sys.executable} -u /content/train_egogeom_omni_v21_relay_g4.py "
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
    print(f"[Launcher] Started detached trainer PID {pid} -> {LOG_FILE}", flush=True)

# Stream log live with periodic heartbeat so websocket stays alive
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
        # Final drain
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            rem = f.read()
            if rem:
                print(rem, end="", flush=True)
        ret = int(EXIT_FILE.read_text().strip() or "1")
        print(f"=== Detached trainer exited with code {ret} ===", flush=True)
        if ret == 0:
            print("=== Flushing Google Drive ===", flush=True)
            try:
                from google.colab import drive
                drive.flush_and_unmount()
                print("Drive cleanly flushed and unmounted!", flush=True)
            except Exception as e:
                print(f"WARN flush: {e}", flush=True)
        else:
            raise SystemExit(ret)
        break
    if not _pid_alive(pid):
        time.sleep(2.0)
        if not EXIT_FILE.exists():
            print(f"=== Trainer PID {pid} ended without exit file ===", flush=True)
            break
    if time.time() - last_out > 45.0:
        print(f"  [heartbeat {time.strftime('%H:%M:%S')}] trainer PID {pid} alive...", flush=True)
        last_out = time.time()
    time.sleep(2.0)
