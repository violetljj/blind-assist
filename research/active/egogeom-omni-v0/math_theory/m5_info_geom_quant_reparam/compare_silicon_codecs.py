"""One-frame matched hardware contrasts for the two Padé range reductions."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from eval_on_v4 import relative_stats
from quant_head import ASSETS, HALF_SPAN, HI, LO, OUT
from silicon_probe import FULL, PROBE

WORK = PROBE.parent
ARMS = {
    "current_base":("matched_codecs","00_rgb_base_rgb",ASSETS/"qnn_v75_v4_final/egogeom_omni_v4_pure_rgb_v75_fp16.bin"),
    "matched_32":("matched_codecs","01_matched_codec_rgb",PROBE/"matched_codec/v4_pure_rgb_final_v75.bin"),
    "matched_4":("matched_codecs","02_matched_codec_r4_rgb",PROBE/"matched_codec_r4/v4_pure_rgb_final_v75.bin"),
    "previous_base":("archive_burst","00_rgb_base_rgb",ASSETS/"qnn_v75_v4_final/egogeom_omni_v4_pure_rgb_v75_fp16.bin"),
    "previous_32":("archive_burst","01_rgb_codec_rgb",OUT/"m5_v4_pure_rgb_v75.bin"),
    "previous_32_repeat":("archive_burst","02_rgb_codec_rgb",OUT/"m5_v4_pure_rgb_v75.bin"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def pairs(left,right,reference):
    ae0,ae1 = np.abs(left-reference),np.abs(right-reference)
    change = 100*(ae1-ae0)/reference
    actual = 100*(right-left)/reference
    return {"from_to_delta_absrel_pp":float(change.mean()),
            "abs_pixel_change_pct_of_fp32":float(np.abs(actual).mean()),
            "pixel_change_abs_p50_p95_p99_pct_of_fp32":np.quantile(np.abs(actual),[.50,.95,.99]).tolist(),
            "improve_tie_worsen_pixels":[int(np.sum(c)) for c in (ae1<ae0,ae1==ae0,ae1>ae0)],
            "identical_depth_pixels":int(np.sum(right==left))}


def level_contingency(condition,target):
    a,ia = np.unique(condition,return_inverse=True)
    b,ib = np.unique(target,return_inverse=True)
    counts = np.bincount(ia.ravel()*len(b)+ib.ravel(),minlength=len(a)*len(b)).reshape(len(a),len(b))
    return {"condition_levels":len(a),"target_levels":len(b),
            "pixels_consistent_with_best_function_of_condition":int(counts.max(axis=1).sum()),
            "levels_with_one_observed_target":int(np.sum((counts>0).sum(axis=1)==1)),
            "max_distinct_target_levels_per_condition":int((counts>0).sum(axis=1).max())}


def inspect_cut(folder,sdk,python):
    table = folder/"compiled_layers.csv"
    if sdk is not None:
        lib = sdk/"lib/x86_64-windows-msvc"
        bindir = sdk/"bin/x86_64-windows-msvc"
        env = dict(os.environ,QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),PYTHONUTF8="1",
                   PYTHONPATH=str(sdk/"lib/python"),PATH=os.pathsep.join(map(str,(lib,bindir,os.environ["PATH"]))))
        with (folder/"compiled_layers.log").open("wb") as stream:
            subprocess.run([str(python),str(bindir/"qairt-dlc-info"),"-i",str(folder/"v4_pure_rgb_final.dlc"),
                            "-s",str(table)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    with table.open(encoding="utf-8",newline="") as stream:
        rows = {r[1]:r for r in csv.reader(stream) if len(r)>=8 and r[0].isnumeric()}
    writer, reader = rows["m5_038_Add"],rows["m5_041_Sub"]
    if not (writer[4] == reader[3] and "Float_16" in writer[4] and "bitwidth 8" in writer[7]):
        raise ValueError(f"Quantizer cut differs in {folder}")
    return {"producer":writer[1],"consumer":reader[1],"connecting_tensor":writer[4],"encoding":writer[7],
            "named_q_or_dq_operator":[k for k in rows if "QuantizeLinear" in k or "DequantizeLinear" in k]}


def compare(sdk=None,python=None):
    with np.load(PROBE/"fp32_taps.npz") as refs:
        reference = refs["depth_m"].astype(float)
    pdir = WORK/"matched_probes/02_matched_taps_rgb/Result_0"
    p = np.fromfile(pdir/"tap_preinv.raw",np.float32).reshape(FULL).astype(float)
    bridge = np.fromfile(pdir/"depth_m.raw",np.float32).reshape(FULL)
    ideal_depth = np.clip(1/p,LO,HI)
    ideal_code = np.clip(np.rint(np.log(ideal_depth/LO)*255/(2*HALF_SPAN)),0,255).astype(int)
    ideal_recon = LO*np.exp(ideal_code*2*HALF_SPAN/255)
    report = {"pixels":reference.size, "unit":"paired pixels of one frozen RGB frame",
              "canonical_input_sha256":{k:digest(ASSETS/"qnn_v75_v4_final"/(k+".raw")) for k in ("rgb","grav_rays")},
              "arms":{}, "contrasts":{},
              "ideal_uint8_after_observed_tap":relative_stats(ideal_recon,reference),
              "ideal_uint8_quantization_only_vs_observed_preinv":relative_stats(ideal_recon,ideal_depth)}
    depths,aux = {},{}
    for label,(run,name,context) in ARMS.items():
        folder = WORK/run/name/"Result_0"
        rec = json.loads((WORK/run/"receipt.json").read_text())
        block, = [b for b in rec["blocks"] if b["block"]==name]
        if block["context_sha256"]!=digest(context) or block["mode"]!="rgb" or block["saved"]!=1:
            raise ValueError(f"Incorrect context or input record at {name}")
        z = np.fromfile(folder/"depth_m.raw",np.float32).reshape(FULL).astype(float)
        assert np.all(np.isfinite(z)) and np.all(z>0)
        depths[label] = z
        aux[label] = {k:np.fromfile(folder/(k+".raw"),np.float32) for k in ("delta_h_m","normal_grav","log_sigma")}
        report["arms"][label] = {
            "context_sha256":block["context_sha256"], "depth_sha256":digest(folder/"depth_m.raw"),
            "htp_ms":block["htp_us"]/1000, "timed_inferences":block["inferences"],
            "distinct_depths":len(np.unique(z)), "distinct_depths_after_rounding_to_fp16":len(np.unique(z.astype(np.float16))),
            "depth_vs_fp32":relative_stats(z,reference),
            "depth_vs_ideal_uint8_of_measured_preinv":relative_stats(z,ideal_recon)}
    # Verify that the measured tap comes from a graph reproducing the uninstrumented base.
    report["tap_bridge_bit_identical_current"] = bool(np.array_equal(bridge,depths["current_base"]))
    if not report["tap_bridge_bit_identical_current"]:
        raise ValueError("Observed pre-reciprocal tap is not bridged to the current base")
    for left,right in (("previous_base","current_base"),("previous_32","previous_32_repeat"),
                       ("previous_32","matched_32"),("matched_32","matched_4"),
                       ("current_base","matched_32"),("current_base","matched_4")):
        entry = pairs(depths[left],depths[right],reference)
        entry["aux_max_abs"] = {k:float(np.abs(aux[right][k]-x).max()) for k,x in aux[left].items()}
        report["contrasts"][left+"_to_"+right] = entry
    report["level_maps"] = {
        "observed_r4_to_r32":level_contingency(depths["matched_4"],depths["matched_32"]),
        "ideal_code_on_measured_base_to_r32":level_contingency(ideal_code,depths["matched_32"]),
        "ideal_code_on_measured_base_to_r4":level_contingency(ideal_code,depths["matched_4"])}
    report["dlc_cut"] = {name:inspect_cut(PROBE/name,sdk,python) for name in ("matched_codec","matched_codec_r4")}
    path = PROBE/"matched_codec_pixel_contrasts.json"
    path.write_text(json.dumps(report,indent=2))
    print([(k,round(v["depth_vs_fp32"]["mean_rel_pct"],5),v["distinct_depths"],v["htp_ms"])
           for k,v in report["arms"].items()])
    for key,entry in report["contrasts"].items():
        print(key,entry["from_to_delta_absrel_pp"],entry["identical_depth_pixels"],entry["aux_max_abs"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qairt-sdk",type=Path)
    ap.add_argument("--qairt-python",type=Path)
    args = ap.parse_args()
    if bool(args.qairt_sdk)!=bool(args.qairt_python):
        ap.error("Supply both QAIRT arguments together to regenerate the DLC layer tables.")
    compare(args.qairt_sdk,args.qairt_python)
