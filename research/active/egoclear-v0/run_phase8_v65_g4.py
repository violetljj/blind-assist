import subprocess, sys, time

print("=== Installing onnx/ort & Launching Phase 8 EgoClear-v6.5 Privileged Distillation (V6-Base -> 57MB Student) ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
t0 = time.time()
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase8_distill_v65_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== EgoClear-v6.5 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    sys.exit(rc)
