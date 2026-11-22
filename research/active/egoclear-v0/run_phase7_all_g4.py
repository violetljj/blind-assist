import subprocess, sys, time

print("=== Phase 7 All-in-One G4: Dual-Cam SANPO-Real 400 Goldmine Pack -> EgoClear-v6-Base (DAV2-Base + 3-Lag 1,2,4) ===", flush=True)
t0 = time.time()
for script in ("/content/pack_phase7_real_dualcam_colab.py", "/content/phase7_train_v6_g4.py"):
    print(f"\n>>> Running {script} ...", flush=True)
    p = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        print(f"!!! {script} FAILED rc={rc}", flush=True)
        sys.exit(rc)
print(f"\n=== Phase 7 All-in-One COMPLETE in {time.time()-t0:.1f}s ===", flush=True)
