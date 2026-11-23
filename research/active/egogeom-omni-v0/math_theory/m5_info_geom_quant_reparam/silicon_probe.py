"""Pure-RGB cutpoint instrumentation and signed pixelwise replay on one archived input."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import onnx
from eval_on_v4 import ASSETS, OUT, qairt_convert, read_inputs, relative_stats, session
from onnx import TensorProto, helper, numpy_helper

PROBE = OUT / "silicon_followup" / "probe"
ORIGINAL_IO = ASSETS / "qnn_v75_v4_final"
NATIVE = ASSETS / "egogeom_omni_v4_best_pure_rgb_fp32.onnx"
S2 = (1, 1, 144, 192)
FULL = (1, 1, 288, 384)
SCALAR = (1, 1, 1, 1)
TAPS = {
    "tap_local": ("/core/disp_head/disp_head.1/Conv_output_0", "Conv", S2),
    "tap_bias": ("/core/global_ctx/Slice_2_output_0", "Slice", SCALAR),
    "tap_logit": ("/core/Add_4_output_0", "Add", S2),
    "tap_canon": ("/core/Softplus_output_0", "Softplus", S2),
    "tap_focal": ("/core/Clip_output_0", "Clip", SCALAR),
    "tap_scaled": ("/core/Clip_1_output_0", "Clip", S2),
    "tap_preinv": ("/core/Clip_2_output_0", "Clip", FULL),
}


def export(target):
    model = onnx.load(NATIVE)
    producer = {key:n for n in model.graph.node for key in n.output}
    rename = {}
    for name, (original, kind, shape) in TAPS.items():
        assert producer[original].op_type == kind
        rename[original] = name
        model.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shape))
    # Only tensor identifiers and declared outputs change. Do not add copies,
    # fake quantizers, or arithmetic to the observed paths.
    for node in model.graph.node:
        for j,x in enumerate(node.input):
            node.input[j] = rename.get(x,x)
        for j,x in enumerate(node.output):
            node.output[j] = rename.get(x,x)
    for vi in model.graph.value_info:
        vi.name = rename.get(vi.name, vi.name)
    onnx.checker.check_model(model)
    onnx.save(model,target)


def resize2(a):
    """ONNX Resize linear, half_pixel, edge replication, scale exactly 2 by 2."""
    h,w = a.shape[-2:]
    result = a.astype(np.float64)
    for axis, size in ((-1,w),(-2,h)):
        coords = np.clip((np.arange(2*size) + .5)/2 - .5, 0, size-1)
        left = np.floor(coords).astype(int)
        right = np.minimum(left+1,size-1)
        f = coords-left
        if axis == -2:
            f = f[:,None]
        result = (1-f)*np.take(result,left,axis=axis) + f*np.take(result,right,axis=axis)
    return result


def local_depth(logit, focal):
    canon = np.logaddexp(0.0,logit.astype(float))
    return canon_depth(canon,focal)


def canon_depth(canon, focal):
    scaled = np.clip(canon.astype(float)*focal.astype(float),0,1/.15)
    return disp_depth(resize2(scaled))


def disp_depth(d):
    return 1/np.clip(d.astype(float),1/25,1/.15)


def ladder(ref, observed):
    """Introduce observed intermediate nodes in topological order, one cut at a time."""
    return {
        "reference_depth":ref["depth_m"].astype(float),
        "local_head":local_depth(observed["tap_local"].astype(float)+ref["tap_bias"], ref["tap_focal"]),
        "global_bias":local_depth(observed["tap_local"].astype(float)+observed["tap_bias"], ref["tap_focal"]),
        "logit_add":local_depth(observed["tap_logit"], ref["tap_focal"]),
        "softplus":canon_depth(observed["tap_canon"], ref["tap_focal"]),
        "focal":canon_depth(observed["tap_canon"], observed["tap_focal"]),
        "multiply_clip":disp_depth(resize2(observed["tap_scaled"])),
        "resize_floor":disp_depth(observed["tap_preinv"]),
        "reciprocal_output":observed["depth_m"].astype(float),
    }


def build(sdk,python):
    PROBE.mkdir(parents=True,exist_ok=True)
    target = PROBE/"rgb_taps.onnx"
    export(target)
    original, probe = session(NATIVE), session(target)
    inputs = read_inputs(original,"rgb",ORIGINAL_IO)
    frozen = dict(zip([o.name for o in original.get_outputs()],original.run(None,inputs)))
    ref = dict(zip([o.name for o in probe.get_outputs()],probe.run(None,inputs)))
    for key, (_, _, shape) in TAPS.items():
        assert ref[key].shape == shape, (key,ref[key].shape,shape)
    parity = {key:float(np.max(np.abs(value-ref[key]))) for key,value in frozen.items()}
    cpu = {name:float(np.max(np.abs(z-ref["depth_m"]))) for name,z in ladder(ref,ref).items()}
    assert max(parity.values()) < 2e-5 and max(cpu.values()) < 2e-5, (parity,cpu)
    np.savez_compressed(PROBE/"fp32_taps.npz",**ref)
    meta = {"onnx_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(NATIVE.read_bytes()).hexdigest(),
            "input_sha256":{key:hashlib.sha256(x.tobytes()).hexdigest() for key,x in inputs.items()},
            "outputs":{key:list(x.shape) for key,x in ref.items()},
            "cpu_vs_native_max_abs":parity, "cpu_reconstruct_depth_max_abs_m":cpu,
            "focal_fp32":float(ref["tap_focal"].item())}
    (PROBE/"host_oracle.json").write_text(json.dumps(meta,indent=2))
    print("CPU",parity,cpu,flush=True)
    if sdk:
        del original,probe
        meta["build"] = qairt_convert(sdk,python,target)
        assert meta["build"]["htp_v75_build"]["verified_target"]["arch"] == 75
        print("HTP",meta["build"],flush=True)
    (PROBE/"host_oracle.json").write_text(json.dumps(meta,indent=2))


def analyze(raw_dir, baseline_raw, output):
    with np.load(PROBE/"fp32_taps.npz") as archive:
        ref = {k:archive[k] for k in ("depth_m", *TAPS)}
    observed = {}
    for k, arr in ref.items():
        # qnn-net-run default converted output raw = float32. Each selected map
        # has one channel, so NCHW and NHWC produce the same file byte order.
        raw = np.fromfile(raw_dir/(k+".raw"),np.float32)
        if raw.size != arr.size or not np.all(np.isfinite(raw)):
            raise ValueError(f"{k}: expected {arr.size} finite float32 values; got {raw.size}")
        observed[k] = raw.reshape(arr.shape)
    baseline = np.fromfile(baseline_raw,np.float32).reshape(FULL)
    recon = ladder(ref,observed)
    z = ref["depth_m"]
    stats, previous = {}, z.astype(float)
    for name,current in recon.items():
        delta = 100*(current-previous)/z
        stats[name] = {
            "depth_vs_fp32":relative_stats(current,z),
            "increment_signed_mean_pct_of_reference":float(delta.mean()),
            "increment_abs_mean_pct_of_reference":float(np.abs(delta).mean()),
            "increment_p05_p50_p95_pct_of_reference":np.quantile(delta,[.05,.5,.95]).tolist(),
        }
        previous = current
    total = stats["reciprocal_output"]["depth_vs_fp32"]["signed_mean_pct"]
    assert abs(sum(v["increment_signed_mean_pct_of_reference"] for v in stats.values())-total)<1e-8
    # Keep raw magnitude diagnostics valid for signed logit/bias tensors near zero.
    taps = {}
    for k, raw in observed.items():
        d = raw.astype(float)-ref[k]
        taps[k] = {"mae":float(np.abs(d).mean()), "signed_mean":float(d.mean()),
                   "rmse":float(np.sqrt(np.mean(d*d))),"max_abs":float(np.abs(d).max())}
        if d.size == 1:
            taps[k].update(fp32=float(ref[k].item()),device=float(raw.item()))
    record = {"pixel_count":z.size, "tap_vs_fp32":taps, "ordered_cut_replays":stats,
              "probe_depth_vs_uninstrumented_device":relative_stats(observed["depth_m"],baseline),
              "probe_depth_bitwise_matches_uninstrumented":bool(np.array_equal(observed["depth_m"],baseline)),
              "uninstrumented_depth_vs_fp32":relative_stats(baseline,z)}
    output.write_text(json.dumps(record,indent=2))
    print(json.dumps({k:(round(a["depth_vs_fp32"]["mean_rel_pct"],5),
                         round(a["increment_abs_mean_pct_of_reference"],5)) for k,a in stats.items()}))
    print("probe vs uninstrumented:",record["probe_depth_vs_uninstrumented_device"])
    return record


def converter_fields(path):
    # QAIRT stores a JSON converterCommand record in the tail part of the DLC.
    with path.open("rb") as stream:
        stream.seek(-min(path.stat().st_size,1_000_000),2)
        segment = stream.read()
    pos = segment.index(b"{",segment.index(b'"converterCommand"'))
    metadata,_ = json.JSONDecoder().raw_decode(segment[pos:].decode("utf-8",errors="ignore"))
    return metadata["args"]


def shorter_decoder(source,destination):
    """Change only x=y/32 -> y/4 and five squarings -> two squarings."""
    model = onnx.load(source)
    reduce = next(x for x in model.graph.initializer if x.name == "m5_reduce")
    assert float(numpy_helper.to_array(reduce)) == 1/32
    reduce.CopyFrom(numpy_helper.from_array(np.array(.25,np.float32),reduce.name))
    squares = [n for n in model.graph.node if n.name.startswith("m5_") and n.op_type=="Mul"
               and n.input[0]==n.input[1]]
    # t^2, x^2 and then the consecutive five final ratio squarings.
    assert len(squares)==7
    last = squares[-5:]
    assert all(b.input[0]==a.output[0] for a,b in itertools.pairwise(last))
    scale = next(n for n in model.graph.node if last[-1].output[0] in n.input)
    scale.input[list(scale.input).index(last[-1].output[0])] = last[1].output[0]
    for node in last[2:]:
        model.graph.node.remove(node)
    onnx.checker.check_model(model)
    onnx.save(model,destination)


def replace_softplus(source, destination):
    """Replace one Softplus with max(x,0) + truncated 2*atanh(e/(2+e))."""
    model = onnx.load(source)
    old = next(node for node in model.graph.node if node.name == "/core/Softplus")
    assert old.op_type == "Softplus"
    insert = list(model.graph.node).index(old)
    nodes = []

    def c(name,value):
        name = "sp_stable_"+name
        model.graph.initializer.append(numpy_helper.from_array(np.array(value,np.float32),name))
        return name

    def op(kind,*inputs,output=None):
        name = "sp_stable_"+str(len(nodes))+"_"+kind
        output = output or name
        nodes.append(helper.make_node(kind,list(inputs),[output],name=name))
        return output

    x = old.input[0]
    t = op("Exp",op("Neg",op("Abs",x)))
    two = c("two",2)
    s = op("Div",t,op("Add",two,t))
    s2 = op("Mul",s,s)
    # 2*s*(1+s^2/3+s^4/5+s^6/7), 0 <= s <= 1/3 for every real x.
    series = c("seventh",1/7)
    for name,coef in (("fifth",1/5),("third",1/3),("one",1)):
        series = op("Add",c(name,coef),op("Mul",s2,series))
    correction = op("Mul",op("Mul",two,s),series)
    op("Add",op("Relu",x),correction,output=old.output[0])
    model.graph.node.remove(old)
    for offset,node in enumerate(nodes):
        model.graph.node.insert(insert+offset,node)
    onnx.checker.check_model(model)
    onnx.save(model,destination)


def matched_build(sdk,python,variant):
    fields = converter_fields(ORIGINAL_IO/"v4_pure_rgb_final.dlc")
    if fields["float_bitwidth"] != "16":
        raise ValueError("Frozen converter precision differs from the expected 16 bits.")
    folder = PROBE / ("matched_"+variant)
    folder.mkdir(parents=True,exist_ok=True)
    named = folder/"v4_pure_rgb_final.onnx"
    if variant in ("codec_r4","softplus"):
        if variant == "codec_r4":
            shorter_decoder(OUT/"m5_v4_pure_rgb.onnx",named)
        else:
            replace_softplus(NATIVE,named)
    else:
        source = {"taps":PROBE/"rgb_taps.onnx", "control":NATIVE,
                  "codec":OUT/"m5_v4_pure_rgb.onnx"}[variant]
        if named.exists() and not named.samefile(source):
            raise FileExistsError(named)
        if not named.exists():
            os.link(source,named)
    args = ["--float_bitwidth","16","--preserve_io_datatype"]
    outputs = ["depth_m","delta_h_m","normal_grav","log_sigma"]
    if variant == "taps":
        outputs.extend(TAPS)
    for io_kind, keys in (("input",["rgb","grav_rays"]),("output",outputs)):
        for key in keys:
            args.extend(["--source_model_"+io_kind+"_layout",key,"NCHW"])
            args.extend(["--desired_"+io_kind+"_layout",key,"NCHW"])
    result = qairt_convert(sdk,python,named,extra_args=args)
    assert result["htp_v75_build"]["verified_target"]["arch"] == 75
    used = converter_fields(named.with_suffix(".dlc"))
    different = {k:{"frozen":fields[k],"new":used.get(k)} for k in fields if fields[k] != used.get(k)}
    # All original inputs and outputs retain the same recipe. Diagnostic outputs
    # necessarily extend the layout lists for the instrumented arm.
    allowed = {"desired_io_layout","output_layout","user_custom_io"} if variant == "taps" else set()
    assert set(different) <= allowed, different
    result["fields_different_from_original"] = different
    result["source_onnx_sha256"] = hashlib.sha256(named.read_bytes()).hexdigest()
    if variant in ("codec","codec_r4","softplus"):
        base = session(NATIVE)
        inp = read_inputs(base,"rgb",ORIGINAL_IO)
        original_values = base.run(None,inp)
        del base
        candidate = session(named)
        estimates = candidate.run(None,inp)
        result["cpu_reference_frame"] = {
            "depth_vs_original_fp32":relative_stats(estimates[0],original_values[0]),
            "auxiliary_max_abs":max(float(np.max(np.abs(a-b))) for a,b in zip(estimates[1:],original_values[1:])),
            "unique_depth_values":len(np.unique(estimates[0]))}
        assert result["cpu_reference_frame"]["auxiliary_max_abs"] < 2e-5
    (folder/"matched_build.json").write_text(json.dumps(result,indent=2))
    print(variant,result["htp_v75_build"],"different fields:",list(different),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    commands = ap.add_subparsers(dest="command",required=True)
    b = commands.add_parser("build")
    b.add_argument("--qairt-sdk",type=Path)
    b.add_argument("--qairt-python",type=Path)
    m = commands.add_parser("build-matched")
    m.add_argument("--qairt-sdk",type=Path,required=True)
    m.add_argument("--qairt-python",type=Path,required=True)
    m.add_argument("--variant",choices=["taps","control","codec","codec_r4","softplus"],required=True)
    a = commands.add_parser("analyze")
    a.add_argument("--raw-dir",type=Path,required=True)
    a.add_argument("--baseline-depth",type=Path,default=ORIGINAL_IO/"out_pure_rgb/depth_m.raw")
    a.add_argument("--out",type=Path,default=PROBE/"probe_device_comparison.json")
    args = ap.parse_args()
    if args.command == "build":
        if bool(args.qairt_sdk) != bool(args.qairt_python):
            ap.error("Supply both QAIRT arguments together.")
        build(args.qairt_sdk,args.qairt_python)
    elif args.command == "build-matched":
        matched_build(args.qairt_sdk,args.qairt_python,args.variant)
    else:
        analyze(args.raw_dir,args.baseline_depth,args.out)
