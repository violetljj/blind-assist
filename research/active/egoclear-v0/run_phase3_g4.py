"""IPython-kernel runner for Phase 3 on G4."""
import subprocess, sys
proc = subprocess.Popen(
    [sys.executable, "-u", "/content/train_and_eval_phase3_g4.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
assert proc.stdout is not None
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"Phase 3 exited with code {rc}")
