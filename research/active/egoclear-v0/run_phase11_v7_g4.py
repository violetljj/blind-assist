import subprocess, sys, time

t0 = time.time()
print("=== [Phase 11 Step 2] Installing onnxconverter-common & launching EgoClear-v7 (MoGe-2 Metric-Normal) Training ===", flush=True)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxconverter-common"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase11_train_v7_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 11 Step 2 EgoClear-v7 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
