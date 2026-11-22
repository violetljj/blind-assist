import subprocess, sys, time

t0 = time.time()
print("=== [Phase 11 v7.5] Installing MoGe + onnx/ort/fp16 & launching EgoClear-v7.5 Training ===", flush=True)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxconverter-common", "git+https://github.com/microsoft/MoGe.git"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase11_train_v75_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 11 EgoClear-v7.5 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
