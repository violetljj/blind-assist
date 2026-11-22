import subprocess, sys, time

print("=== Launching Phase 5b Calibrated EgoClear-v4.1 (6-thread staging + 15% indoor + slope-aware hyst) ===", flush=True)
t0 = time.time()
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase5b_calibrated_v41_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 5b exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    sys.exit(rc)
