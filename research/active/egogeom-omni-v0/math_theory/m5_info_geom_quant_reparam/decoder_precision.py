"""Assess a 256-code depth decoder independently of encoder and backbone errors."""

import argparse
import json
import zipfile

import numpy as np
from quant_head import CENTER, HALF_SPAN, HI, LO, OUT
from silicon_probe import ORIGINAL_IO, PROBE, converter_fields


def rounded_pade(code, reduction, degree):
    f = np.float16
    y = code.astype(f)*f(2*HALF_SPAN/255)-f(HALF_SPAN)
    x = y/f(reduction)
    x2 = x*x
    if degree == 2:
        even = f(12)+x2
        odd = f(6)*x
    elif degree == 3:
        even = f(120)+f(12)*x2
        odd = x*(f(60)+x2)
    else:
        raise ValueError(degree)
    result = (even+odd)/(even-odd)
    for _ in range(reduction.bit_length()-1):
        result = result*result
    return (f(CENTER)*result).clip(f(LO),f(HI)).astype(float)


def run(raw_path):
    old = ORIGINAL_IO/"v4_pure_rgb_final.dlc"
    codec = OUT/"m5_v4_pure_rgb.dlc"
    old_fields, new_fields = converter_fields(old), converter_fields(codec)
    differences = {k:{"frozen":old_fields.get(k),"codec":new_fields.get(k)}
                   for k in sorted(set(old_fields)|set(new_fields)) if old_fields.get(k)!=new_fields.get(k)}
    config = {"differences":differences}
    for k,p in (("original",old),("codec",codec)):
        with zipfile.ZipFile(p) as dlc:
            config[k+"_dlc_payload_bytes"] = {e.filename:e.file_size for e in dlc.infolist()}
    (PROBE/"original_vs_codec_converter_fields.json").write_text(json.dumps(config,indent=2))
    code = np.arange(256)
    ideal = CENTER*np.exp(code*(2*HALF_SPAN/255)-HALF_SPAN)
    outputs = {"current_pade_2_2_scale32":rounded_pade(code,32,2),
               "pade_2_2_scale4":rounded_pade(code,4,2),
               "pade_3_3_scale2":rounded_pade(code,2,3),
               "ideal_decoder_rounded_to_fp16":ideal.astype(np.float16).astype(float)}
    if raw_path:
        device = np.fromfile(raw_path,np.float32)
        if not len(device) or not np.all(np.isfinite(device)):
            raise ValueError("Expected nonempty finite float32 device depth")
    report = {}
    for name,depth in outputs.items():
        error = 100*np.abs(depth/ideal-1)
        stats = {"distinct_depths_for_256_codes":len(np.unique(depth)),
                 "absrel_pct_on_uniform_codes":float(error.mean()),"worst_rel_pct_on_codes":float(error.max())}
        if raw_path:
            unique,counts = np.unique(device,return_counts=True)
            # An output can agree with some code even when the true device code differs.
            # The nearest-code residual is a lower bound with codes hidden.
            nearest = np.min(100*np.abs(unique[:,None].astype(float)/depth[None,:]-1),axis=1)
            stats["observed_unique_depths"] = len(unique)
            stats["levels_with_exact_match_to_some_code"] = int(np.sum(nearest==0))
            stats["pixel_weighted_mean_nearest_code_rel_pct"] = float(np.average(nearest,weights=counts))
            stats["worst_nearest_code_rel_pct"] = float(nearest.max())
        report[name] = stats
    (PROBE/"decoder_precision.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--device-depth",type=Path)
    run(ap.parse_args().device_depth)
