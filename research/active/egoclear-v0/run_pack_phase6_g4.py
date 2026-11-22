import subprocess, sys, time

print("=== Launching Phase 6 48-vCPU Treasure Data Packer (444 SANPO-Real + 210 ARKitScenes) ===", flush=True)
t0 = time.time()
p = subprocess.Popen(
    [sys.executable, "-u", "/content/pack_phase6_treasure_colab.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 6 Packer exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    sys.exit(rc)
