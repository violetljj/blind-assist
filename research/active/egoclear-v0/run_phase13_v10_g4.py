import subprocess, sys, time

t0 = time.time()
print("=== [Phase 13 Step 2] Launching EgoClear-v10 (Mega-Data + Isotropic Border-Mask + 3D Homography + Unlocked MoGe-2 Neck/Last2 + DAV2 Aux) ===", flush=True)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase13_train_v10_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 13 EgoClear-v10 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
