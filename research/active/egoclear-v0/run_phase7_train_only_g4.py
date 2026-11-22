import subprocess, sys, time

print("=== Installing onnx/onnxruntime & Launching EgoClear-v6-Base (1,194 scenes on 96 GB G4) ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
t0 = time.time()
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase7_train_v6_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== EgoClear-v6-Base exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    sys.exit(rc)
