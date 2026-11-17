# On-phone SNPE timing for UniDepth-V2-S (camera-baked) DLC: sh phone_bench.sh <HxW> <tag> [--use_gpu --gpu_mode float16 | --use_dsp]
cd /data/local/tmp/udepth
export LD_LIBRARY_PATH=/data/local/tmp/udepth/lib
export ADSP_LIBRARY_PATH="/data/local/tmp/udepth/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
S=$1; TAG=$2; shift 2
rm -rf out_${S}_${TAG}
./snpe-net-run --container unidepth_s_cam_${S}_fp32.dlc --input_list list21_$S.txt "$@" --perf_profile burst --output_dir out_${S}_${TAG} > log_${S}_${TAG}.txt 2>&1
echo "$S $TAG rc=$?"; ls out_${S}_${TAG} | head -3
