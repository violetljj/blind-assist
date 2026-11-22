import subprocess, sys, time

print("=== Launching Phase 9 Step 2: EgoClear-v6.6 Dual-Teacher Clean-KD + 5-Domain & 4-Model CODa Eval ===", flush=True)
t0 = time.time()
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase9_v66_cleankd_and_coda_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 9 Step 2 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    sys.exit(rc)
