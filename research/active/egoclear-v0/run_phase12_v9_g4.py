import subprocess, sys, time

t0 = time.time()
print("=== [Phase 12] Launching EgoClear-v9 (Dual-Geometry + VIO TTC + Dual-Champion v6.6+v8.5 KD + ViT-L & GT Scale De-Bias + Pairwise Rank) ===", flush=True)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase12_train_v9_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 12 EgoClear-v9 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
