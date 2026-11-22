import subprocess, sys, time

t0 = time.time()
print("=== [Phase 11 Stage 2] Launching 4-Gate Diagnostics on Frozen EgoClear-v7.5 ===", flush=True)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase11_stage2_diagnostics_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 11 Stage 2 Diagnostics exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
