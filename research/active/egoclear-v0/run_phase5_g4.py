import subprocess, sys, time

print("=== Launching Phase 5 Unified Indoor+Outdoor EgoClear-v4 + Event-Level Hysteresis ===", flush=True)
t0 = time.time()
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase5_unified_indoor_hysteresis_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 5 exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    sys.exit(rc)
