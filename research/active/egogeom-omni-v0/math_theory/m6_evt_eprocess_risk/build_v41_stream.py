"""Materialize the pure-RGB v4.1 observations with the frozen v4 camera/BEV pipeline."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
import build_v4_stream as shared
from egogeom_omni_v41_model import EgoGeomOmniV41Net, EgoGeomV41Config
import numpy as np
import psutil
import torch


def verify_completed(root, checkpoint_hash):
    """Verify both the archived observations and their unchanged original inputs."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    receipt = json.loads((root.parent / HERE.name / "v5_empirical_report_v41.json").read_text(encoding="utf8"))
    if manifest["checkpoint_sha256"] != checkpoint_hash or shared.digest(root/"manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("v4.1 manifest or checkpoint differs from the scored run")
    observed = set()
    hashes = {k.replace("\\", "/"): v for k, v in receipt["stream_sha256"].items()}
    for suite, (folder, shade, _, expected) in shared.SUITES.items():
        sources = sorted((shared.WORK / "zeroshot_eval" / folder).rglob("*.npz"))
        entries = {e["name"]:e for e in manifest["suites"][suite]["clips"]}
        if len(sources) != expected or {p.stem for p in sources} != set(entries):
            raise ValueError(f"{suite}: changed source list")
        for source in sources:
            path = root / suite / source.name
            key = path.relative_to(root).as_posix()
            digest = shared.digest(source)
            if digest != entries[source.stem]["input_sha256"] or shared.digest(path) != hashes[key]:
                raise ValueError(f"{suite}/{source.name}: content differs from the scored run")
            observed.add(key)
            with np.load(path) as pack:
                tag = f"{checkpoint_hash}:{manifest['protocol_sha256']}:{manifest['batch']}:{digest}:{shade}"
                if str(pack["identity"]) != tag:
                    raise ValueError(f"{key}: cached recipe mismatch")
    if len(observed) != len(receipt["stream_sha256"]):
        raise ValueError("extra files in scored stream receipt")
    print(f"Verified {len(observed)} archived v4.1 streams against the scored receipt")


def main():
    checkpoint = shared.WORK / "egogeom_omni_v41_pure_rgb_best.pt"
    root = shared.ROOT.parent / "v41_stream"
    batch = 8
    checkpoint_hash = shared.digest(checkpoint)
    if (root/"manifest.json").is_file() and (root.parent/HERE.name/"v5_empirical_report_v41.json").is_file():
        verify_completed(root, checkpoint_hash)
        return
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.75)
    shared.cv2.setNumThreads(1)
    dependencies = [Path(__file__), Path(shared.__file__), *[shared.ROUTE / src for src in (
        "eval_egogeom_v3_zeroshot.py", "eval_cross_glasses_zeroshot.py", "egogeom_temporal_bev.py",
        "egogeom_omni_v41_model.py", "egogeom_omni_v4_model.py", "egogeom_losses_and_physics.py")]]
    code_hash = hashlib.sha256("".join(shared.digest(f) for f in dependencies).encode()).hexdigest()
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = EgoGeomOmniV41Net(EgoGeomV41Config(**state["cfg"]), pretrained_backbone=False)
    # The P1 checkpoint carries an older dormant ToF branch. Match every RGB key strictly.
    model.tof_branch = None
    model.load_state_dict({k: v for k, v in state["ema"].items() if not k.startswith("tof_branch.")}, strict=True)
    del state
    model.cuda().eval()
    archive = json.loads((shared.WORK / "cross_glasses_zeroshot_v41_final.json").read_text())
    manifest = dict(protocol_sha256=code_hash, checkpoint_sha256=checkpoint_hash, batch=batch,
                    device=torch.cuda.get_device_name(), precision="BF16 autocast, PyTorch RGB eval path",
                    suites={})
    for suite in ("aria_gen2", "coda_campus", "arkit_val30_day", "arkit_val30_dark018"):
        folder, shade, cutoff, expected = shared.SUITES[suite]
        sources = sorted((shared.WORK / "zeroshot_eval" / folder).rglob("*.npz"))
        assert len(sources) == len({p.stem for p in sources}) == expected
        dest = root / suite
        dest.mkdir(parents=True, exist_ok=True)
        clips, entries = [], []
        for i, source in enumerate(sources, 1):
            shared.host_headroom()
            input_hash = shared.digest(source)
            tag = f"{checkpoint_hash}:{code_hash}:{batch}:{input_hash}:{shade}"
            path = dest / source.name
            if not path.exists():
                obs = shared.observe(model, source, suite, shade, batch)
                with path.with_suffix(".partial").open("wb") as f:
                    np.savez(f, identity=tag, **obs)
                os.replace(path.with_suffix(".partial"), path)
                del obs
            with np.load(path) as pack:
                assert str(pack["identity"]) == tag
                clip = {k: pack[k] for k in ("d_1f_wz", "d_bev_v23", "r_hyb_v23")}
            with np.load(source) as source_data:
                clip.update(eval_rows=source_data["eval_rows"], name=source.stem)
            clips.append(clip)
            entries.append(dict(name=source.stem, frames=len(clip["d_1f_wz"]), input_sha256=input_hash))
            if i == 1 or i % 10 == 0 or i == expected:
                print(suite, i, expected, "GPU reserved MiB", torch.cuda.max_memory_reserved()//2**20,
                      "free RAM MiB", psutil.virtual_memory().available//2**20, flush=True)
        metrics = {"1f_world_z": shared.score_dataset(clips, "d_1f_wz", None, cutoff),
                   "stage2_bev_v23_hyb": shared.score_dataset(clips, "d_bev_v23", "r_hyb_v23", cutoff)}
        manifest["suites"][suite] = dict(clips=entries, metrics=metrics, archive=archive["suites"][suite])
    tmp = root / "manifest.partial"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    os.replace(tmp, root / "manifest.json")


if __name__ == "__main__":
    main()
