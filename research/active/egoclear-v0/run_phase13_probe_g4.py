import subprocess, sys, time

t0 = time.time()
print("=== [Phase 13 Step 1] Launching Aria 1.762x Isotropic Frontend & Failure Autopsy Probe ===", flush=True)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase13_aria_isotropic_probe_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 13 Probe exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
