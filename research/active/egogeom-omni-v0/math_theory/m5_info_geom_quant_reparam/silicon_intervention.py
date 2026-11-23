"""Cross the Softplus and focal interventions on frozen S24 Ultra inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from eval_on_v4 import qairt_convert, read_inputs, session
from onnx import TensorProto, helper
from silicon_panel import metric
from silicon_probe import ASSETS, NATIVE, PROBE, converter_fields
from silicon_run import IO, WORK, digest


def focal_input(directory):
    rays = np.fromfile(directory / "grav_rays.raw", "<f4").reshape(1, 3, 288, 384)
    du = np.diff(rays, axis=-1)
    norm = np.sqrt(np.sum(du * du, axis=1) + np.float32(1e-12))
    return np.clip(norm.max() * np.float32(350), .2, 4).astype("<f4").reshape(1, 1, 1, 1)


def prepare():
    dest = WORK / "focal_inputs"
    dest.mkdir(exist_ok=True)
    panel = WORK / "panel_inputs"
    rows = json.loads((panel / "frames.json").read_text())
    for tag, folder in [("archived", IO), *((r["id"], panel / r["id"]) for r in rows)]:
        scalar = focal_input(folder)
        scalar.tofile(dest / (tag + ".raw"))
    with np.load(PROBE / "fp32_taps.npz") as taps:
        np.testing.assert_allclose(focal_input(IO), taps["tap_focal"], rtol=2e-7)


def make_graph(source, target):
    model = onnx.load(source)
    g = model.graph
    key = "/core/Clip_output_0"
    users = [n for n in g.node if key in n.input]
    assert len(users) == 1 and users[0].op_type == "Mul"
    users[0].input[list(users[0].input).index(key)] = "host_focal"
    g.input.append(helper.make_tensor_value_info("host_focal", TensorProto.FLOAT, [1, 1, 1, 1]))
    # Drop only the branch disconnected by this intervention, including its constants.
    needed = {o.name for o in g.output}
    dead_nodes = []
    for node in reversed(g.node):
        if needed.intersection(node.output):
            needed.update(node.input)
        else:
            dead_nodes.append(node)
    for node in dead_nodes:
        g.node.remove(node)
    for initializer in list(g.initializer):
        if initializer.name not in needed:
            g.initializer.remove(initializer)
    for value in list(g.value_info):
        if value.name not in needed:
            g.value_info.remove(value)
    onnx.checker.check_model(model)
    onnx.save(model, target)


def build(arm, sdk, python):
    source = NATIVE if arm == "focal" else PROBE / "matched_softplus/v4_pure_rgb_final.onnx"
    dest = WORK / "factorial" / arm
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "v4_pure_rgb_final.onnx"
    make_graph(source, target)
    ref = session(source)
    inp = read_inputs(ref, "rgb", IO)
    old = ref.run(None, inp)
    del ref
    candidate = session(target)
    out = candidate.run(None, dict(inp, host_focal=focal_input(IO)))
    errs = [float(np.max(np.abs(a-b))) for a, b in zip(out, old, strict=True)]
    assert max(errs) < 2e-5, errs
    recipe = ["--float_bitwidth", "16", "--preserve_io_datatype"]
    for kind, ports in (("input", candidate.get_inputs()), ("output", candidate.get_outputs())):
        for port in ports:
            recipe.extend([f"--source_model_{kind}_layout", port.name, "NCHW",
                           f"--desired_{kind}_layout", port.name, "NCHW"])
    del candidate
    report = qairt_convert(sdk, python, target, extra_args=recipe)
    assert report["htp_v75_build"]["verified_target"]["arch"] == 75, report
    fields = converter_fields(IO / "v4_pure_rgb_final.dlc")
    actual = converter_fields(target.with_suffix(".dlc"))
    diffs = {k for k, v in fields.items() if v != actual.get(k)}
    assert diffs <= {"input_layout", "desired_io_layout", "user_custom_io"}, diffs
    report.update(source_sha256=digest(source), onnx_sha256=digest(target),
                  changed_recipe_keys=sorted(diffs), max_abs_vs_source_fp32=errs)
    (dest / "build.json").write_text(json.dumps(report, indent=2))
    print(arm, errs, report["htp_v75_build"])


def analyze(folder, base_dir):
    def blocks(root):
        receipt = json.loads((root / "receipt.json").read_text())
        return {r["arm"]: root / r["block"] for r in reversed(receipt["blocks"])}
    paths = {"rgb_base": blocks(base_dir)["rgb_base"], **blocks(folder)}
    is_panel = paths["rgb_base"].name.endswith("_panel")
    panel = WORK / "panel_inputs"
    rows = json.loads((panel / "frames.json").read_text()) if is_panel else [{"id": "archived"}]
    ref = session(NATIVE)
    results = []
    references = []
    for i, row in enumerate(rows):
        inputs = read_inputs(ref, "rgb", panel / row["id"] if is_panel else IO)
        depth = ref.run(["depth_m"], inputs)[0].ravel()
        references.append(depth)
        record = dict(row)
        for arm in ("rgb_base", "matched_softplus", "focal", "both"):
            raw = np.fromfile(paths[arm] / f"Result_{i}/depth_m.raw", "<f4")
            assert raw.shape == depth.shape and np.isfinite(raw).all() and raw.min() > 0
            record[arm] = metric(raw, depth)
            record[arm]["matching_aux_files"] = sum(
                digest(paths[arm] / f"Result_{i}" / key) == digest(paths["rgb_base"] / f"Result_{i}" / key)
                for key in ("delta_h_m.raw", "normal_grav.raw", "log_sigma.raw")
            )
        results.append(record)
    del ref
    for arm, src in [("matched_softplus", PROBE / "matched_softplus"), ("focal", WORK / "factorial/focal"),
                     ("both", WORK / "factorial/both")]:
        cpu = session(src / "v4_pure_rgb_final.onnx")
        for row, record, reference in zip(rows, results, references, strict=True):
            directory = panel / row["id"] if is_panel else IO
            inp = {p.name: np.fromfile(directory / (p.name + ".raw"), "<f4").reshape(p.shape)
                   for p in cpu.get_inputs() if p.name != "host_focal"}
            if arm in ("focal", "both"):
                inp["host_focal"] = focal_input(directory)
            host = cpu.run(["depth_m"], inp)[0].ravel()
            record[arm]["host_vs_original_max_abs_pct"] = float(100*np.max(np.abs(host/reference-1)))
        del cpu
    (folder / "factorial_fidelity.json").write_text(json.dumps(results, indent=2))
    for arm in ("rgb_base", "matched_softplus", "focal", "both"):
        print(arm, np.mean([r[arm]["mean_abs_pct"] for r in results]))


def analyze_transfer(folder, rgb_folder):
    receipt = json.loads((folder / "receipt.json").read_text())
    blocks = {(r["arm"],r["mode"]):folder / r["block"] / "Result_0"
              for r in reversed(receipt["blocks"])}
    rgb_receipt = json.loads((rgb_folder / "receipt.json").read_text())
    rgb_block = next(r["block"] for r in rgb_receipt["blocks"] if r["arm"] == "both")
    ref = session(ASSETS / "egogeom_omni_v4_best_omni_fp32.onnx")
    results = {}
    for mode in ("zero", "cnh", "dep"):
        host = ref.run(["depth_m"], read_inputs(ref, mode, IO))[0].ravel()
        base, both = [np.fromfile(blocks[arm, mode] / "depth_m.raw", "<f4")
                      for arm in ("omni_base", "omni_both")]
        assert host.shape == base.shape == both.shape and np.isfinite(both).all() and both.min() > 0
        aux = ("delta_h_m.raw", "normal_grav.raw", "log_sigma.raw")
        results[mode] = {"base":metric(base,host), "both":metric(both,host),
                         "identical_aux":sum(digest(blocks["omni_base",mode]/k) ==
                                             digest(blocks["omni_both",mode]/k) for k in aux)}
        if mode == "zero":
            rgb = np.fromfile(rgb_folder/rgb_block/"Result_0/depth_m.raw","<f4")
            results[mode]["rgb_both_vs_zero_max_abs_m"] = float(np.abs(rgb-both).max())
    (folder / "transfer.json").write_text(json.dumps(results,indent=2))
    print(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    b = sub.add_parser("build")
    b.add_argument("--arm", choices=["focal", "both"], required=True)
    b.add_argument("--sdk", type=Path, required=True)
    b.add_argument("--python", type=Path, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--run", required=True)
    a.add_argument("--base", required=True)
    t = sub.add_parser("transfer")
    t.add_argument("--run", required=True)
    t.add_argument("--rgb-run", required=True)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "build":
        build(args.arm, args.sdk, args.python)
    elif args.command == "transfer":
        analyze_transfer(WORK / args.run, WORK / args.rgb_run)
    else:
        analyze(WORK / args.run, WORK / args.base)
