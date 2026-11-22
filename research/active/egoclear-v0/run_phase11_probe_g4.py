import subprocess, sys, time

t0 = time.time()
print("=== [Phase 11 Step 1] Installing onnx & launching MoGe-2-ViT-S-Normal 4-Domain Probe ===", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase11_moge2_probe_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 11 Step 1 Probe exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
