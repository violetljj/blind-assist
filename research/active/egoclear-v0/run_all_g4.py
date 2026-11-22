"""IPython-kernel runner for G4: Phase 0 multi-core pack -> Phase 1 baselines & EgoClear -> Drive flush."""
import os
import subprocess
import sys

print(f"[g4-runner] vCPUs={os.cpu_count()}, drive_ok={os.path.isdir('/content/drive/MyDrive/BlindAssist')}", flush=True)

for script in ("/content/pack_sanpo_colab.py", "/content/train_and_eval_g4.py"):
    print(f"\n[g4-runner] === Running {script} ===", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{script} exited with code {rc}")

try:
    from google.colab import drive
    drive.flush_and_unmount()
    print("[g4-runner] drive.flush_and_unmount() complete!", flush=True)
except Exception as e:
    print(f"[g4-runner] flush note: {e}", flush=True)
