import subprocess, sys
proc = subprocess.Popen([sys.executable, "-u", "/content/finish_phase4_eval_onnx_g4.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"finisher exited with {rc}")
try:
    from google.colab import drive
    drive.flush_and_unmount()
    print("[p4-fin-runner] drive.flush_and_unmount() complete!", flush=True)
except Exception as e:
    print(f"[p4-fin-runner] flush note: {e}", flush=True)
