"""Build two frozen ONNX-17 codecs, exercise their actual Q/DQ, replay the device frame."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from quant_head import (
    ASSETS,
    CENTER,
    HALF_SPAN,
    HI,
    LO,
    OUT,
    gauss_pairs,
    int_log_roundtrip,
)


def export(source, destination):
    model = onnx.load(source)
    graph = model.graph
    assert any(o.version == 17 and not o.domain for o in model.opset_import)
    original = {out:node for node in graph.node for out in node.output}
    clip = original["depth_m"]
    mult = original[clip.input[0]]
    recip = next(original[i] for i in mult.input if original[i].op_type == "Reciprocal")
    assert clip.op_type == "Clip" and mult.op_type == "Mul" and original[recip.input[0]].op_type == "Clip"
    # Keep terminal physical floor/ceiling; replace only the single-user reciprocal path.
    for node in (recip, mult):
        assert sum(node.output[0] in n.input for n in graph.node) == 1
    for node in (recip, mult, clip):
        graph.node.remove(node)
    index = 0

    def constant(name, value, dtype=np.float32):
        name = "m5_" + name
        graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=dtype), name))
        return name

    def op(kind, *inputs, output=None):
        nonlocal index
        index += 1
        name = f"m5_{index:03d}_{kind}"
        output = output or name
        graph.node.append(helper.make_node(kind, inputs, [output], name=name))
        return output

    one = constant("one", 1)
    center = constant("center", CENTER)
    half = constant("half_span", HALF_SPAN)
    cd = op("Mul", recip.input[0], center)
    t = op("Div", op("Sub", one, cd), op("Add", one, cd))
    t2 = op("Mul", t, t)
    terms = []
    for j, (q, w) in enumerate(gauss_pairs()):
        denom = op("Sub", one, op("Mul", t2, constant(f"q{j}", q)))
        terms.append(op("Div", constant(f"w{j}", w), denom))
    # Balanced summation avoids a serial reduction across eight maps.
    while len(terms) > 1:
        terms = [op("Add", terms[i], terms[i+1]) for i in range(0,len(terms),2)]
    positive_log = op("Add", op("Mul", t, terms[0]), half)
    scale = constant("scale", 2*HALF_SPAN/255)
    zero = constant("zero", 0, dtype=np.uint8)
    code = op("QuantizeLinear", positive_log, scale, zero)
    read = op("DequantizeLinear", code, scale, zero)
    x = op("Mul", op("Sub", read, half), constant("reduce", 1/32))
    even = op("Add", constant("twelve", 12), op("Mul", x, x))
    odd = op("Mul", x, constant("six", 6))
    ratio = op("Div", op("Add", even, odd), op("Sub", even, odd))
    for _ in range(5):
        ratio = op("Mul", ratio, ratio)
    op("Clip", op("Mul", ratio, center), clip.input[1], clip.input[2], output="depth_m")
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return {"opset":17, "new_nodes":index, "terminal_source":recip.input[0],
            "terminal_code_dtype":"UINT8", "original_nodes_replaced":[recip.name, mult.name, clip.name]}


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def read_inputs(runtime, mode, directory):
    names = {"rgb":"rgb", "grav_rays":"grav_rays", "tof_grid_s4":f"tof_{mode}",
             "has_tof":"ht_zero" if mode == "zero" else "ht_one"}
    return {i.name:np.fromfile(directory/(names[i.name]+".raw"), np.float32).reshape(i.shape)
            for i in runtime.get_inputs()}


def exhaustive_output_cut(path, source_input):
    # Enumerate every normal FP32 code in the positive allowed disparity range.
    # Extract the actual ONNX nodes: repeated evaluations of the backbone add no
    # information about the output transfer function at a prescribed disparity.
    model = onnx.load(path)
    nodes = [n for n in model.graph.node if n.name.startswith("m5_")]
    internal = {o for n in nodes for o in n.output}
    external = {i for n in nodes for i in n.input} - internal - {source_input}
    constants = [n for n in model.graph.node if n.op_type == "Constant" and external.intersection(n.output)]
    initializers = [v for v in model.graph.initializer if v.name in external]
    graph = helper.make_graph(constants+nodes, "m5_cut", [
        helper.make_tensor_value_info(source_input,TensorProto.FLOAT,[None])], [
        helper.make_tensor_value_info("depth_m",TensorProto.FLOAT,[None])], initializers)
    cut = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    onnx.checker.check_model(cut)
    cut_path = OUT/"m5_cut.onnx"
    onnx.save(cut,cut_path)
    del model
    engine = session(cut_path)
    first, last = [int(np.float32(x).view(np.uint32)) for x in (1/HI,1/LO)]
    worst, witness = -1.0, 0.0
    for begin in range(first,last+1,262144):
        d = np.arange(begin,min(last+1,begin+262144),dtype=np.uint32).view(np.float32)
        z = engine.run(["depth_m"],{source_input:d})[0]
        reference = np.clip(1/d.astype(float),LO,HI)
        loss = np.abs(z/reference-1)
        j = np.argmax(loss)
        if loss[j] > worst:
            worst,witness = float(loss[j]),float(d[j])
    assert worst < .0102
    return {"count_fp32_codes":last-first+1,"max_rel_pct":100*worst,"disparity_at_max":witness,
            "backend":"ONNX Runtime CPU, actual elementwise output nodes, flat tensor"}


def relative_stats(value, reference):
    signed = 100*(value.astype(float)-reference)/reference
    err = np.abs(value.astype(float)-reference)
    rel = np.abs(signed)
    return {"mae_mm":float(1e3*np.mean(err)), "mean_rel_pct":float(np.mean(rel)),
            "max_rel_pct":float(np.max(rel)), "signed_mean_pct":float(signed.mean()),
            "signed_p05_p50_p95_pct":np.quantile(signed, [.05,.5,.95]).tolist()}


def far_field_stats(device, fp32):
    far = fp32 >= 12
    rel = np.abs(device.astype(float) / fp32 - 1)
    return {"fp32_ge12m_pixels":int(far.sum()), "total_pixels":int(far.size),
            "share_of_sum_absrel_pct":float(100*rel[far].sum()/rel.sum()) if rel.sum() else None}


def read_context_info(binary_dir, binary, env):
    info_path = binary.with_suffix(".info.json")
    subprocess.run([str(binary_dir/"qnn-context-binary-utility.exe"), "--context_binary",str(binary),
                    "--json_file",str(info_path), "--unified_qairt_format"], env=env, check=True)
    info = json.loads(info_path.read_text())
    target = {"arch":info["contextMetadata"]["dspArch"], "soc":info["socModel"],
              "graph":info["graphs"][0]["graphName"], "info":info_path.name}
    if (target["arch"], target["soc"]) != (75,57):
        raise ValueError(f"Unexpected target: {target}")
    return target


def qairt_convert(sdk, python, source):
    binary_dir = sdk / "bin/x86_64-windows-msvc"
    env = dict(os.environ, QAIRT_SDK_ROOT=str(sdk), QNN_SDK_ROOT=str(sdk), PYTHONUTF8="1",
               PYTHONPATH=str(sdk/"lib/python"), PYTHONDONTWRITEBYTECODE="1")
    env["PATH"] = os.pathsep.join(map(str,[sdk/"lib/x86_64-windows-msvc", binary_dir, env["PATH"]]))
    dlc = source.with_suffix(".dlc")
    cmd = [str(python), str(binary_dir/"qairt-converter"), "--input_network", str(source),
           "--output_path", str(dlc)]
    log = source.with_suffix(".converter.log")
    with log.open("wb") as stream:
        result = subprocess.run(cmd, cwd=OUT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    converted = {"exit_code":result.returncode, "log":log.name,
                 "dlc_bytes":dlc.stat().st_size if result.returncode == 0 and dlc.is_file() else 0}
    if converted["dlc_bytes"]:
        lib = sdk/"lib/x86_64-windows-msvc"
        base_cfg = ASSETS/"qnn_v75_v4_final/htp_rgb.json"
        config = json.loads(base_cfg.read_text())
        config["graphs"][0]["graph_names"] = [source.stem]
        htp_config = source.with_suffix(".htp.json")
        htp_config.write_text(json.dumps(config), encoding="utf-8")
        ext = source.with_suffix(".ext.json")
        ext.write_text(json.dumps({"backend_extensions":{
            "shared_library_path":str(lib/"QnnHtpNetRunExtensions.dll"),
            "config_file_path":str(htp_config)}}), encoding="utf-8")
        htp_log = source.with_suffix(".htp.log")
        binary = source.with_name(source.stem + "_v75.bin")
        cmd = [str(binary_dir/"qnn-context-binary-generator.exe"), "--backend",str(lib/"QnnHtp.dll"),
               "--model",str(lib/"QnnModelDlc.dll"), "--dlc_path",str(dlc),
               "--config_file",str(ext), "--binary_file",binary.stem, "--output_dir",str(OUT)]
        with htp_log.open("wb") as stream:
            build = subprocess.run(cmd, cwd=OUT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        converted["htp_v75_build"] = {"exit_code":build.returncode, "log":htp_log.name,
                                     "bytes":binary.stat().st_size if build.returncode == 0 and binary.is_file() else 0}
        if converted["htp_v75_build"]["bytes"]:
            converted["htp_v75_build"]["verified_target"] = read_context_info(binary_dir,binary,env)
    return converted


def run(sdk=None, python=None):
    OUT.mkdir(parents=True, exist_ok=True)
    io = ASSETS/"qnn_v75_v4_final"
    frozen = json.loads((ASSETS/"s24u_v4_final_npu_report.json").read_text())["fidelity"]
    report = {"source":"one archived S24U input frame, four modalities, CPU ONNX Runtime",
              "frozen_s24u_fidelity":frozen, "models":{}}
    for graph, modes in (("pure_rgb", ["rgb"]), ("omni", ["zero","cnh","dep"])):
        src = ASSETS / f"egogeom_omni_v4_best_{graph}_fp32.onnx"
        target = OUT / f"m5_v4_{graph}.onnx"
        entry = export(src, target)
        if graph == "pure_rgb":
            report["full_range_fp32_onnx_uint8_cut"] = exhaustive_output_cut(target,entry["terminal_source"])
        entry["source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
        entry["onnx_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        entry["archived_frame"] = {}
        baseline, candidate = session(src), session(target)
        for mode in modes:
            inp = read_inputs(baseline, mode, io)
            base, *aux = baseline.run(None, inp)
            estimate, *new_aux = candidate.run(None, inp)
            folder = "out_pure_rgb" if graph == "pure_rgb" else f"out_omni_{mode}"
            npu = np.fromfile(io/folder/"depth_m.raw", np.float32).reshape(base.shape)
            # Idealized cut holds the whole FP32 graph upstream fixed.
            tail = np.reciprocal(np.reciprocal(base.astype(float)).astype(np.float16).astype(float))
            tail = tail.astype(np.float16).astype(float).clip(LO,HI)
            ideal = int_log_roundtrip(1/base.astype(float)).clip(LO,HI)
            record = {"pixels":int(base.size), "fp32_range_m":[float(base.min()),float(base.max())],
                      "candidate_vs_fp32":relative_stats(estimate, base),
                      "archived_device_vs_ort_fp32":relative_stats(npu, base),
                      "device_far_field":far_field_stats(npu, base),
                      "fp16_terminal_only_vs_fp32":relative_stats(tail, base),
                      "archived_device_vs_terminal_residual_pct_of_fp32":float(100*np.mean(abs(npu-tail)/base)),
                      "candidate_vs_float64_codec":relative_stats(estimate, ideal),
                      "auxiliary_max_abs":max(float(abs(a-b).max()) for a,b in zip(aux,new_aux))}
            assert record["candidate_vs_fp32"]["max_rel_pct"] < 1.02 and record["auxiliary_max_abs"] < 1e-4
            entry["archived_frame"][mode] = record
        del baseline, candidate
        if sdk is not None:
            entry["qairt_conversion"] = qairt_convert(sdk,python,target)
        report["models"][graph] = entry
        print(graph, entry["archived_frame"], entry.get("qairt_conversion"), flush=True)
    (OUT/"v4_empirical_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qairt-sdk", type=Path)
    parser.add_argument("--qairt-python", type=Path)
    args = parser.parse_args()
    if bool(args.qairt_sdk) != bool(args.qairt_python):
        parser.error("Supply both QAIRT arguments together.")
    run(args.qairt_sdk, args.qairt_python)
