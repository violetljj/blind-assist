# On-phone SNPE timing for DAV2-Small DLC: sh dav2s_phone_bench.sh <HxW> <tag> [--use_gpu --gpu_mode float16 | --use_dsp]
# Push snpe-net-run, libSNPE/libSnpeHtp{Prepare,V75Stub,V75CalculatorStub}.so to lib/, libSnpeHtpV75Skel.so to dsp/ (QAIRT 2.47), DLC, input raw and list21_<HxW>.txt first.
cd /data/local/tmp/dav2s
export LD_LIBRARY_PATH=/data/local/tmp/dav2s/lib
export ADSP_LIBRARY_PATH="/data/local/tmp/dav2s/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
S=$1; TAG=$2; shift 2
rm -rf out_${S}_${TAG}
./snpe-net-run --container dav2s_${S}_fp32.dlc --input_list list21_$S.txt "$@" --perf_profile burst --output_dir out_${S}_${TAG} > log_${S}_${TAG}.txt 2>&1
echo "$S $TAG rc=$?"; ls out_${S}_${TAG} | head -3
