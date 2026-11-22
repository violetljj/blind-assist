"""IPython-kernel runner for Phase 2 on G4: pack +450 synth & 40 real -> saturate G4 training & 3-regime eval -> flush Drive."""
import os
import subprocess
import sys

print(f"[p2-runner] vCPUs={os.cpu_count()}, drive_ok={os.path.isdir('/content/drive/MyDrive/BlindAssist')}", flush=True)

for script in ("/content/pack_phase2_colab.py", "/content/train_and_eval_phase2_g4.py"):
    print(f"\n[p2-runner] === Running {script} ===", flush=True)
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

print("[p2-runner] Phase 2 complete (Drive kept mounted for overnight Phase 3 iteration; will flush before stop).", flush=True)
