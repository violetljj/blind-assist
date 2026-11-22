import subprocess, sys, time

t0 = time.time()
print("=== Launching Phase 13 48-vCPU Turbo Mega-Data Factory ===", flush=True)
p = subprocess.Popen(
    [sys.executable, "-u", "/content/pack_phase13_megapack_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Mega-Data Factory exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    raise SystemExit(rc)
