"""Build the frozen Omni graph with stable Softplus and a host-computed focal factor."""

import argparse
import ast
import io
import json
from pathlib import Path

import numpy as np
from eval_on_v4 import qairt_convert, read_inputs, relative_stats, session
from quant_head import ASSETS
from silicon_intervention import focal_input, make_graph
from silicon_probe import ORIGINAL_IO, PROBE, converter_fields, replace_softplus
from silicon_run import digest


def build(sdk,python):
    folder = PROBE.parent/"factorial/omni_both"
    folder.mkdir(parents=True,exist_ok=True)
    source = ASSETS/"egogeom_omni_v4_best_omni_fp32.onnx"
    onnx_path = folder/"v4_omni_final.onnx"
    with io.BytesIO() as intermediate:
        replace_softplus(source,intermediate)
        intermediate.seek(0)
        make_graph(intermediate,onnx_path)
    archive = converter_fields(ORIGINAL_IO/"v4_omni_final.dlc")
    if archive["float_bitwidth"] != "16" or archive["preserve_io_datatype"] != "[[]]":
        raise ValueError("Unexpected frozen Omni precision recipe")
    original = session(source)
    input_sets,expected = {},{}
    for mode in ("zero","cnh","dep"):
        input_sets[mode] = read_inputs(original,mode,ORIGINAL_IO)
        expected[mode] = original.run(None,input_sets[mode])
    del original
    candidate = session(onnx_path)
    scalar = focal_input(ORIGINAL_IO)
    report = {"source_sha256":digest(source),"onnx_sha256":digest(onnx_path),
              "host_focal":scalar.item(), "reference_pixels_per_mode":288*384,"cpu_vs_original":{}}
    out_names = [x.name for x in candidate.get_outputs()]
    for mode,inp in input_sets.items():
        actual = candidate.run(None,dict(inp,host_focal=scalar))
        aux = {k:float(np.max(np.abs(a-b))) for k,a,b in zip(out_names[1:],actual[1:],expected[mode][1:],strict=True)}
        stats = relative_stats(actual[0],expected[mode][0])
        assert max(aux.values()) < 2e-5 and stats["max_rel_pct"] < .01, (mode,aux,stats)
        report["cpu_vs_original"][mode] = {"depth":stats,"aux_max_abs":aux}
    inputs = [[x.name,"NCHW"] for x in candidate.get_inputs()]
    outputs = [[x.name,"NCHW"] for x in candidate.get_outputs()]
    del candidate
    assert inputs == ast.literal_eval(archive["input_layout"])+[["host_focal","NCHW"]]
    assert outputs == ast.literal_eval(archive["output_layout"])
    args = ["--float_bitwidth","16","--preserve_io_datatype"]
    for kind,ports in (("input",inputs),("output",outputs)):
        for name,layout in ports:
            args.extend(["--source_model_"+kind+"_layout",name,layout,"--desired_"+kind+"_layout",name,layout])
    print("CPU reference passed",[(k,v["depth"]["max_rel_pct"]) for k,v in report["cpu_vs_original"].items()],flush=True)
    result = qairt_convert(sdk,python,onnx_path,extra_args=args)
    compiled = result["htp_v75_build"]["verified_target"]
    assert (compiled["arch"],compiled["soc"],compiled["graph"]) == (75,57,"v4_omni_final")
    fields = converter_fields(onnx_path.with_suffix(".dlc"))
    diff = [k for k,v in archive.items() if fields.get(k)!=v]
    assert set(diff) <= {"input_layout","desired_io_layout","user_custom_io"}, diff
    report.update(build=result,changed_recipe_keys=diff,inputs_nchw=inputs,outputs_nchw=outputs)
    (folder/"build.json").write_text(json.dumps(report,indent=2))
    print("Compiled",compiled,result["htp_v75_build"]["bytes"],flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qairt-sdk",type=Path,required=True)
    ap.add_argument("--qairt-python",type=Path,required=True)
    args = ap.parse_args()
    build(args.qairt_sdk,args.qairt_python)
