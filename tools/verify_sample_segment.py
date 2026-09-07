"""Verify a captured WillowSampleV1 run and materialize sanitized replay inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research/active/dtr-r0/unreal"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded(root, relative):
    require(not Path(relative).is_absolute(), "Input paths must be relative")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root.resolve()), "Input path escapes its input root")
    return path


def verify(run):
    report = {"status": "FAIL", "authority": "DEVELOPMENT_SAMPLE_ENGINEERING_ONLY",
              "file_hashes": {}, "native_status": "NOT_CHECKED", "proxy_status": "NOT_CHECKED",
              "sanitized_contract_pass": False, "rgbd_pair_count": 0, "view_dimensions": []}
    try:
        import ue_dtr_replay as replay
        receipt = read(run / "receipt.json")
        require(receipt["status"] == "PASS", "Builder receipt is not PASS")
        shader_failures = [line for line in (run / "editor.log").read_text(encoding="utf-8", errors="replace").splitlines()
                           if any(namespace in line for namespace in ("SampleMaterialsV2", "SampleFinish"))
                           and "Failed to compile Material" in line]
        require(not shader_failures, "Sample material compilation failed; default fallback is not acceptable")
        report["sample_material_compile_errors"] = 0
        require(receipt["source_unchanged"] is True, "Builder reports source mutation")
        require(receipt["map"] == "/Game/StreetLab/WillowSampleV1", "Unsupported sample source map")
        project = REPO / "artifacts.local/unreal/BlindAssistStreetLab/Content/StreetLab"
        require(sha(project / "StreetLabV4.umap") == receipt["source_sha256"], "Source map hash changed")
        require(sha(project / "WillowSampleV1.umap") == receipt["map_sha256"], "Saved sample map hash changed")
        report["source_unchanged"] = True
        report["map_sha256"] = receipt["map_sha256"]
        geometry = read(run / "evaluator/static_geometry.json")
        require(geometry["schema"] == "willow_sample_native_aabb_v1", "Unsupported geometry contract")
        require(geometry["coverage"]["export_completeness"] ==
                "ENUMERATED_ALL_LOADED_STATIC_MESH_COMPONENTS_AND_INSTANCES", "Incomplete native geometry declaration")
        from sample_segment_contract import evaluate_route
        witnesses = read(run / "evaluator/witnesses.json")
        comparisons = {}
        for name in ("center", "left_bypass", "right_bypass", "contact_control"):
            witness = witnesses[name]
            hits = witness["native_capsule_hit_actors"]
            require(len(hits) == len(witness["path_world_xy_m"]) - 1 and len(hits) > 0,
                    f"Missing native segment sweeps: {name}")
            require(all(hit is None or isinstance(hit, str) and bool(hit) for hit in hits),
                    f"Invalid native hit data: {name}")
            native_contact = any(hit is not None for hit in hits)
            proxy = evaluate_route(witness["path_world_xy_m"], geometry)
            require(proxy == witness["proxy"], f"Stored proxy differs from recomputation: {name}")
            comparisons[name] = {"native_contact": native_contact, "native_hit_actors": hits,
                                 "proxy_status": proxy["status"], "proxy_contact": proxy["contact"],
                                 "agreement": native_contact == proxy["contact"]}
        report["witnesses"] = comparisons
        report["proxy_status"] = "AGREES_WITH_NATIVE_WITNESSES" if all(
            value["agreement"] for value in comparisons.values()) else "PROXY_NATIVE_DISCREPANCIES"
        report["proxy_limit"] = "Conservative native AABBs are not triangle truth; canopy/aggregate bounds may intersect free native corridors."
        native_ok = all(not comparisons[name]["native_contact"] for name in ("center", "left_bypass", "right_bypass")) and comparisons["contact_control"]["native_contact"]
        report["native_status"] = "PASS" if native_ok else "FAIL"
        require(native_ok, "Native clear witnesses or seeded contact control failed")
        expected_views=0 if receipt.get('capture_mode')=='sensors_only' else 4
        require(len(receipt["views"]) == expected_views, "View count differs from capture mode")
        for view in receipt["views"]:
            path = bounded(run, view["path"])
            with Image.open(path) as image:
                image.load()
                expected=tuple(receipt.get('view_resolution',[1920,1080]))
                require(expected in ((1920,1080),(3840,2160)) and image.format == "PNG" and image.size == expected,
                        "Showcase PNG must match its declared native resolution")
                report["view_dimensions"].append({"path": view["path"], "width": image.width, "height": image.height})
        model = run / "model"
        sensors = read(model / "sensor_manifest.json")
        replay.adapter.assert_sanitized_model_value(sensors)
        require(set(sensors) == {"calibration", "frames", "route_origin_world_m", "authority"}, "Unexpected sensor manifest keys")
        require(sensors["authority"] == "SENSOR_ONLY_RGB_FORWARD_DEPTH_AND_EGO_POSES", "Unsupported sensor authority")
        calibration = sensors["calibration"]
        require(calibration == {"width": 640, "height": 360, "horizontal_fov_degrees": 100., "depth_max_m": 100.}, "Unsupported RGB-D calibration")
        frames = sensors["frames"]
        require(len(frames) == 11, "Expected 11 RGB-D frames")
        origin = sensors["route_origin_world_m"]
        require(len(origin) == 3 and np.isfinite(origin).all(), "Invalid route origin")
        clean_frames, depths = [], []
        frame_keys = {"sample_index", "time_s", "rgb_path", "depth_path", "camera_transform", "wearer_transform", "command_velocity"}
        for index, frame in enumerate(frames):
            require(set(frame) == frame_keys, "Unexpected sensor frame keys")
            require(frame["sample_index"] == index and abs(frame["time_s"] - index * .1) < 1e-7, "Frame indices/timestamps do not match 10 Hz capture")
            for key in ("camera_transform", "wearer_transform"):
                require(set(frame[key]) == {"x", "y", "z", "pitch", "yaw", "roll"} and np.isfinite(list(frame[key].values())).all(), "Invalid calibrated pose")
            velocity = frame["command_velocity"]
            require(set(velocity) == {"x", "y", "z"} and np.isfinite(list(velocity.values())).all(), "Invalid ego command")
            with Image.open(bounded(model, frame["rgb_path"])) as image:
                image.load()
                require(image.format == "PNG" and image.size == (640, 360), "RGB dimensions/format mismatch")
            depth = np.load(bounded(model, frame["depth_path"]), allow_pickle=False)
            require(depth.dtype == np.float32 and depth.shape == (360, 640), "Depth dimensions/dtype mismatch")
            require(np.isfinite(depth).all() and (depth >= 0).all() and (depth <= 100).all(), "Invalid depth values")
            if 'sensor_eye_height_m' in receipt:
                eye=frame['camera_transform']['z']-frame['wearer_transform']['z']
                require(abs(eye-1.70)<1e-6 and abs(eye-receipt['sensor_eye_height_m'])<1e-6,
                        'First-person optical center must be 1.70 m above the floor')
                # Known clear center-floor patch is an independent depth check
                # of optical height; this result remains evaluator-only.
                focal=640/(2*np.tan(np.deg2rad(calibration['horizontal_fov_degrees']/2)))
                pitch=np.deg2rad(frame['camera_transform']['pitch'])
                rows=np.arange(280,331)+.5-180
                dz=np.sin(pitch)-rows[:,None]/focal*np.cos(pitch)
                measured=float(np.median(-depth[280:331,300:341]*dz))
                require(abs(measured-eye)<.02, 'Ground-depth optical-height check exceeds 2 cm')
                report.setdefault('eye_height_checks',[]).append({'frame':index,'pose_height_m':eye,'depth_height_m':measured})
            fraction = float(np.mean((depth > 0) & (depth < 100)))
            require(fraction > .2, "Insufficient valid depth fraction")
            depths.append({"sample_index": index, "valid_fraction": fraction})
            timestamp, pose = frame["time_s"], frame["wearer_transform"]
            plan = {"schema_version": "dtr-c1-plan-receipt-v1", "coordinate_frame": "ANCHOR_FORWARD_RIGHT",
                    "plan_id": f"sample-{index}", "session_id": "sample", "issued_at_s": timestamp,
                    "valid_from_s": timestamp, "expires_at_s": timestamp + 4.,
                    "time_parameterized_waypoints": [{"time_s": round(timestamp + j * .2, 5),
                        "forward_m": pose["x"] - origin[0] + velocity["x"] * j * .2,
                        "right_m": pose["y"] - origin[1] + velocity["y"] * j * .2} for j in range(21)]}
            plan["receipt_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
            relative = f"sample/plan-{index:04d}.json"
            write(model / relative, plan)
            clean_frames.append(dict(frame, plan_path=relative))
        episode = {"episode_id": "sample", "route_frame": {"center_xy_m": origin[:2], "z_origin_m": origin[2],
                   "forward_xy": [1, 0], "right_xy": [0, 1]}, "plan_path": clean_frames[0]["plan_path"], "frames": clean_frames}
        write(model / "manifest.json", {"schema_version": "ue-sample-segment-v1", "calibration": calibration, "episodes": [episode]})
        contract = replay.load_contract(model)
        require(len(contract.episodes) == 1 and len(contract.episodes[0].observations) == 11, "Sanitized replay count mismatch")
        report.update(sanitized_contract_pass=True, rgbd_pair_count=11, depth_checks=depths,
                      plan_authority="Issued navigation derived only from sensor ego poses and command velocities", status="PASS")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for path in sorted(run.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".json", ".png", ".npy") and path.name != "verification.json":
                report["file_hashes"][path.relative_to(run).as_posix()] = sha(path)
        write(run / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    require(run.is_relative_to((REPO / "artifacts.local").resolve()), "Run must be under artifacts.local")
    report = verify(run)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
