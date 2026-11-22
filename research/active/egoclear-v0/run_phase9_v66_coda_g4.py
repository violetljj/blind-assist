"""Master Phase 9 G4 runner:
1. Ensure onnx/onnxruntime/pyyaml installed.
2. Pack UT Austin CODa 24-clip zero-shot external campus benchmark via ZIP64 HTTP Range (or restore from Drive).
3. Run Dual-Teacher Complementary Clean-KD EgoClear-v6.6 + 5-domain eval + 4-model CODa zero-shot benchmark + FP16 ONNX export.
"""
import subprocess
import sys
import time

t0 = time.time()
print("=== [Phase 9] Installing onnx/ort/pyyaml ===", flush=True)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "pyyaml"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)

print("=== [Phase 9 Step 1/2] Packing UT Austin CODa Zero-Shot External Campus Benchmark ===", flush=True)
p1 = subprocess.run([sys.executable, "-u", "/content/pack_coda_zeroshot_colab.py"])
print(f"=== CODa packer exited rc={p1.returncode} in {time.time()-t0:.1f}s ===", flush=True)
if p1.returncode != 0:
    raise SystemExit(p1.returncode)

t1 = time.time()
print("=== [Phase 9 Step 2/2] Training EgoClear-v6.6 Dual-Teacher Clean-KD + 5-Domain & 4-Model CODa Eval ===", flush=True)
p2 = subprocess.run([sys.executable, "-u", "/content/phase9_v66_cleankd_and_coda_g4.py"])
print(f"=== Phase 9 EgoClear-v6.6 + CODa exited rc={p2.returncode} in {time.time()-t1:.1f}s (total {time.time()-t0:.1f}s) ===", flush=True)
if p2.returncode != 0:
    raise SystemExit(p2.returncode)
