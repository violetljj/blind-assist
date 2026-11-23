"""Export a real one-byte log-disparity port and decode it with a 256-entry table."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from eval_on_v4 import qairt_convert, read_inputs, session
from onnx import TensorProto, helper, numpy_helper
from quant_head import ASSETS, HI, LO
from silicon_intervention import focal_input
from silicon_panel import metric
from silicon_run import IO, WORK, digest

BYTE = WORK / "byte"
SOURCE = ASSETS / "egogeom_omni_v4_best_pure_rgb_fp32.onnx"
STEP = np.log(HI / LO) / 255
LUT = np.asarray(HI * np.exp(-STEP * np.arange(256)), dtype="<f4")


def build(sdk, python, corrected=False):
    folder = WORK / "byte_both" if corrected else BYTE
    folder.mkdir(exist_ok=True, parents=True)
    source = WORK / "factorial/both/v4_pure_rgb_final.onnx" if corrected else SOURCE
    model = onnx.load(source)
    graph = model.graph
    owner = {key:n for n in graph.node for key in n.output}
    clip = owner["depth_m"]
    mul = owner[clip.input[0]]
    inv = next(owner[k] for k in mul.input if owner[k].op_type == "Reciprocal")
    assert owner[inv.input[0]].op_type == "Clip" and clip.op_type == "Clip"
    for node in (inv, mul):
        assert sum(node.output[0] in n.input for n in graph.node) == 1
    for node in (inv,mul,clip):
        graph.node.remove(node)
    graph.output.remove(next(x for x in graph.output if x.name=="depth_m"))
    graph.output.insert(0,helper.make_tensor_value_info("depth_code",TensorProto.UINT8,[1,1,288,384]))
    for key,val in [("m5_max",np.float32(HI)),("m5_step",np.float32(STEP)),("m5_zero",np.uint8(0))]:
        graph.initializer.append(numpy_helper.from_array(np.array(val),key))
    for op,inputs,key in [("Mul",[inv.input[0],"m5_max"],"m5_ratio"),
                          ("Log",["m5_ratio"],"m5_logratio"),
                          ("QuantizeLinear",["m5_logratio","m5_step","m5_zero"],"depth_code")]:
        graph.node.append(helper.make_node(op,inputs,[key],name=key))
    onnx.checker.check_model(model)
    target = folder/("head_byte_both.onnx" if corrected else "head_byte.onnx")
    onnx.save(model,target)
    del model
    LUT.tofile(folder/"decode_f32.raw")
    original = session(SOURCE)
    inputs = read_inputs(original,"rgb",IO)
    ref = original.run(["depth_m"],inputs)[0]
    del original
    encoder = session(target)
    if corrected:
        inputs["host_focal"] = focal_input(IO)
    codes = encoder.run(["depth_code"],inputs)[0]
    assert codes.dtype == np.uint8 and codes.size == ref.size
    host_err = float((np.abs(LUT[codes]/ref-1)).max()*100)
    assert host_err < 1.02
    del encoder
    recipe = ["--float_bitwidth","16","--preserve_io_datatype"]
    for io,keys in [("input",list(inputs)),("output",["depth_code","delta_h_m","normal_grav","log_sigma"])]:
        for key in keys:
            recipe.extend([f"--source_model_{io}_layout",key,"NCHW",f"--desired_{io}_layout",key,"NCHW"])
    build_log = qairt_convert(sdk,python,target,extra_args=recipe)
    report = {"source_sha256":digest(source),"onnx_sha256":digest(target),"lut_sha256":digest(folder/"decode_f32.raw"),
              "host_frame_max_absrel_pct":host_err,"code_bytes":codes.nbytes,"build":build_log}
    if build_log.get("htp_v75_build",{}).get("bytes"):
        meta = json.loads(target.with_name(target.stem+"_v75.info.json").read_text())
        entry = next(o for o in meta["graphs"][0]["graphOutputs"] if o["name"]=="depth_code")
        report["native_code_port"] = {k:entry[k] for k in ("name","dataType","dimensions","quantizeParams")}
    (folder/"build.json").write_text(json.dumps(report,indent=2))
    assert report.get("native_code_port",{}).get("dataType") in ("QNN_DATATYPE_UINT_8","QNN_DATATYPE_UFIXED_POINT_8"),report
    print(json.dumps(report["native_code_port"]))


def analyze(run_dir, base_dir, unpacked_dir=None):
    current = json.loads((run_dir/"receipt.json").read_text())
    base_receipt = json.loads((base_dir/"receipt.json").read_text())
    run = next(b for b in current["blocks"] if b["arm"] in ("byte","byte_both"))
    corrected = run["arm"] == "byte_both"
    folder = WORK / "byte_both" if corrected else BYTE
    base = next(b for b in base_receipt["blocks"] if b["arm"]=="rgb_base")
    unpacked = None
    if corrected:
        if unpacked_dir is None:
            raise ValueError("Supply the paired uncompressed corrected run")
        unpacked = next(b for b in json.loads((unpacked_dir/"receipt.json").read_text())["blocks"]
                        if b["arm"]=="both")
        assert unpacked["mode"] == run["mode"] and unpacked["saved"] == run["saved"]
    assert run["mode"] == base["mode"] and run["saved"] == base["saved"]
    if not np.array_equal(LUT,np.fromfile(folder/"decode_f32.raw","<f4")):
        raise ValueError("Decoder table mismatch")
    is_panel = run["mode"]=="panel"
    panel = WORK/"panel_inputs"
    rows = json.loads((panel/"frames.json").read_text()) if is_panel else [{"id":"archived"}]
    original = session(SOURCE)
    encoder = session(folder/("head_byte_both.onnx" if corrected else "head_byte.onnx"))
    records = []
    for i,row in enumerate(rows):
        inputs = read_inputs(original,"rgb",panel/row["id"] if is_panel else IO)
        ref = original.run(["depth_m"],inputs)[0].ravel()
        if corrected:
            inputs["host_focal"] = focal_input(panel/row["id"] if is_panel else IO)
        host = encoder.run(["depth_code"],inputs)[0].ravel()
        native = np.fromfile(run_dir/run["block"]/f"Result_{i}/depth_code_native.raw","u1")
        baseline = np.fromfile(base_dir/base["block"]/f"Result_{i}/depth_m.raw","<f4")
        assert len(ref)==len(host)==len(native)==len(baseline)==288*384
        records.append({**row,"native_bytes":native.nbytes,"levels_device":len(np.unique(native)),
                        "base_vs_fp32":metric(baseline,ref),"byte_vs_fp32":metric(LUT[native],ref),
                        "host_byte_vs_fp32":metric(LUT[host],ref),"byte_vs_device_base":metric(LUT[native],baseline),
                        "byte_vs_fp32_max_abs_pct":float(100*np.abs(LUT[native]/ref-1).max())})
        if unpacked:
            full = np.fromfile(unpacked_dir/unpacked["block"]/f"Result_{i}/depth_m.raw","<f4")
            assert full.shape == native.shape
            records[-1]["byte_vs_corrected_unpacked"] = metric(LUT[native],full)
    (run_dir/"byte_fidelity.json").write_text(json.dumps(records,indent=2))
    print({k:np.mean([r[k]["mean_abs_pct"] for r in records]) for k in ("base_vs_fp32","byte_vs_fp32","byte_vs_device_base")})


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="action",required=True)
    b=sub.add_parser("build")
    b.add_argument("--sdk",type=Path,required=True)
    b.add_argument("--python",type=Path,required=True)
    b.add_argument("--corrected",action="store_true")
    e=sub.add_parser("analyze")
    e.add_argument("--run",required=True)
    e.add_argument("--base",required=True)
    e.add_argument("--unpacked",help="Uncompressed corrected run on the same inputs")
    args=ap.parse_args()
    if args.action=="build":
        build(args.sdk,args.python,args.corrected)
    else:
        analyze(WORK/args.run,WORK/args.base,WORK/args.unpacked if args.unpacked else None)
