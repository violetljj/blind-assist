import subprocess, sys, time

t0 = time.time()
print("=== [Phase 11 Stage 3B] Launching EgoClear-v8.5 (Pure B=16 Unified MoGe-2 + lags=(1,2) + ViT-L Geom KD + Clean Dual-Teacher KD) ===", flush=True)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase11_train_v85_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 11 EgoClear-v8.5 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
