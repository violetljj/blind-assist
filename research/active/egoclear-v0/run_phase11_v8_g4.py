import subprocess, sys, time

t0 = time.time()
print("=== [Phase 11 Stage 3] Launching EgoClear-v8 (lags=1,2 + 18ch Physics Shortcut + 35% RGB Drop + ViT-L Residual KD + Clear/Dyn Floor) ===", flush=True)
p = subprocess.Popen(
    [
        sys.executable, "-u", "/content/phase11_train_v8_g4.py",
        "--auto-from-stage2",
        "--warm-start-v75", "0",
        "--epochs", "14",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 11 EgoClear-v8 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
