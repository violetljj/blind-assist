#!/usr/bin/env bash
# Track A scale v4 pipeline (frozen protocol CNH_TRACK_A_SCALE_V4_PROTOCOL_20260926.md).
# usage: run_scale_v4.sh smoke|formal   -- stops at the first failed hard gate (formal).
set -u
MODE=$1
cd /e/linnan/linnan/research/active/dtr-r0/nearfield
PY=/e/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe
if [ "$MODE" = formal ]; then
  FAM=cnh-track-a-scale-v4-20260926; N=96; SC="0 32 64"; CONC=$(seq 32 5 95)
else
  FAM=cnh-track-a-scale-v4-smoke-20260926; N=4; SC="0 2 2"; CONC="2"
fi
E=E:/linnan/linnan/artifacts.local/evidence/$FAM-v1
EB=/e/linnan/linnan/artifacts.local/evidence/$FAM-v1
mkdir -p $EB/geometry $EB/sensor $EB/readouts-gpu $EB/analysis/visibility $EB/analysis/ceiling
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
stop() { echo "[$(date +%T)] STOP: $1"; [ "$MODE" = formal ] && exit 2; }

echo "[$(date +%T)] $MODE geometry start ($N units, split $SC)"
seq 0 $((N-1)) | xargs -P 17 -I{} sh -c 'u=$(printf %02d {}); '$PY' -u cnh_track_a_v13_generate.py --output '$E'/geometry/unit$u --units {} --family '$FAM' --split-counts '"$SC"' --fast-margin --no-10hz > '$E'/geometry/log-unit$u.txt 2>&1'
echo "[$(date +%T)] geometry done; repair"
$PY cnh_track_a_v13_repair.py --root $E/geometry --n-units $N --family $FAM --split-counts $SC --fast-margin || stop repair
$PY cnh_track_a_v13_merge.py --root $E/geometry --n-units $N --v2-gates --split-counts $SC > $EB/geometry/merge.txt 2>&1
head -c 600 $EB/geometry/merge.txt; echo
grep -q '"status": "GEOMETRY_READY"' $EB/geometry/merge.txt || stop "geometry gate"

echo "[$(date +%T)] sensor start (mount -10 only)"
seq 0 $((N-1)) | xargs -P 17 -I{} sh -c 'uu=$(printf %02d {}); [ -f '$E'/geometry/unit$uu/unit$uu.json ] || exit 0; '$PY' cnh_track_a_v13_sensor.py --geometry '$E'/geometry --output '$E'/sensor --unit {} --mount -10 --family '$FAM' --rate 5 > '$E'/sensor/log-unit$uu-mount-10.txt 2>&1'
echo "[$(date +%T)] sensor done"
$PY cnh_track_a_scale_g3g4.py $E $N -10 > $EB/sensor/g3g4.txt 2>&1
head -c 400 $EB/sensor/g3g4.txt; echo
grep -q '"G3": true, .*"G4": true' $EB/sensor/g3g4.txt || stop "G3/G4"

echo "[$(date +%T)] GPU readouts start (4 workers); CPU concordance runs alongside"
$PY cnh_track_a_scale_gpu_run.py --geometry $E/geometry --sensor $E/sensor --root $E/readouts-gpu --family $FAM \
  --split-counts $SC --workers 4 --conditions primary-mount-10-snr6:-10:6 > $EB/readouts-gpu/log.txt 2>&1 &
GPU=$!
until [ -f $EB/readouts-gpu/primary-mount-10-snr6/bias.npy ] || ! kill -0 $GPU 2>/dev/null; do sleep 2; done
$PY cnh_track_a_v4_concordance.py --evidence $E --units $CONC --family $FAM --workers 6 > $EB/readouts-gpu/concordance.txt 2>&1 &
CPU=$!
wait $GPU; echo "[$(date +%T)] GPU readouts done (exit $?)"; tail -n 2 $EB/readouts-gpu/log.txt
wait $CPU; echo "[$(date +%T)] concordance done"; cat $EB/readouts-gpu/concordance.txt
grep -q '"pass_gate": true' $EB/readouts-gpu/concordance.txt || stop "GPU/CPU concordance"

R=$E/readouts-gpu/primary-mount-10-snr6
echo "[$(date +%T)] descriptive diagnostics start"
seq $(echo $SC | awk '{print $1+$2}') $((N-1)) | xargs -P 4 -I{} $PY cnh_memory_visibility_diagnostic.py --geometry $E/geometry --sensor $E/sensor --readouts $R --units {} --family $FAM --out $E/analysis/visibility > /dev/null &
V=$!
seq $(echo $SC | awk '{print $1+$2}') $((N-1)) | xargs -P 4 -I{} $PY cnh_signal_ceiling_diagnostic.py --geometry $E/geometry --sensor $E/sensor --readouts $R --units {} --family $FAM --out $E/analysis/ceiling > /dev/null &
C=$!
wait $V $C
echo "[$(date +%T)] analysis start"
$PY cnh_track_a_v4_analysis.py $R $E/analysis/visibility $E/analysis/ceiling $E/analysis/v4_analysis.json > $EB/analysis/log.txt 2>&1 || stop analysis
echo "[$(date +%T)] $MODE DONE"
