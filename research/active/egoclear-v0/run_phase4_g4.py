"""IPython-kernel runner for Phase 4 on G4 + final Drive flush_and_unmount()."""
import subprocess, sys
proc = subprocess.Popen(
    [sys.executable, "-u", "/content/phase4_simreal_aria_onnx_g4.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
assert proc.stdout is not None
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"Phase 4 exited with code {rc}")
try:
    from google.colab import drive
    drive.flush_and_unmount()
    print("[p4-runner] drive.flush_and_unmount() complete!", flush=True)
except Exception as e:
    print(f"[p4-runner] flush note: {e}", flush=True)
