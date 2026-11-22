import subprocess, sys, time

t0 = time.time()
print("=== [Phase 12 Probe] Installing MoGe & running EgoClear-v9 Architecture Probe ===", flush=True)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "git+https://github.com/microsoft/MoGe.git"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase12_v9_probe_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 12 Probe exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
