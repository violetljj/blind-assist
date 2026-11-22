import subprocess, sys, time

t0 = time.time()
print("=== [Phase 14] Launching EgoClear-v10.5 (Autoseg Guard + Dyn Ranking + Floor-Free OHEM + Pixel-Aligned KD + Causal Head-Scan + IsoGravCrop + 2x DAV2 Aux + EMA) ===", flush=True)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase14_train_v105_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 14 EgoClear-v10.5 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
