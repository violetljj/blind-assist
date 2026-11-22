import subprocess, sys, time

print("=== Launching Phase 6 Cloud Autopsy + FP16/INT8 Quant + MP4 Replay on G4 ===", flush=True)
t0 = time.time()
p = subprocess.Popen(
    [sys.executable, "-u", "/content/phase6_cloud_autopsy_quant_replay_g4.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"=== Phase 6 Cloud Autopsy+Edge exited rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
if rc != 0:
    sys.exit(rc)
