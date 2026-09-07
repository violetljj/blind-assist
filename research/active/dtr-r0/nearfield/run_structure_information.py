"""One-time NF-G6 prediction, followed by native-only offline evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from run_ground_anchor import ROOT
from rgb_replay import _metric_class
from sparse_structure import match, pose_matrix
from diagnose_depth_loss import distribution, sha
from research_backend import torch_observation


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def predict(capture, output, source, weights):
    manifest_path=capture/"model/sensor_manifest.json"
    manifest=read(manifest_path)
    frames=manifest["frames"]
    assert len(frames)==36 and len({f["clip_id"] for f in frames})==12
    assert read(capture/"receipt.json")["status"]=="PASS"
    assert all("depth_path" not in f for f in frames)
    cls,identity=_metric_class(source)
    assert weights.name=="depth_anything_v2_metric_hypersim_vits.pth"
    model=cls(encoder="vits",features=64,out_channels=[48,96,192,384],max_depth=20.)
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    model.to("cuda:0").eval()
    rows=[]
    with torch.inference_mode():
        for start in range(0,36,3):
            window=frames[start:start+3]
            assert len({f["clip_id"] for f in window})==1
            assert [f["frame_in_clip"] for f in window]==[0,1,2]
            paths=[capture/"model"/f["rgb_path"] for f in window]
            images=[cv2.imread(str(p),cv2.IMREAD_COLOR) for p in paths]
            assert all(i is not None and i.shape==(360,640,3) for i in images)
            poses=[f["camera_transform"] for f in window]
            torch.cuda.synchronize();tick=time.perf_counter()
            sparse=match(images,poses)
            torch.cuda.synchronize();match_ms=(time.perf_counter()-tick)*1000
            tick=time.perf_counter()
            dense=np.asarray(model.infer_image(images[-1],518),dtype=np.float32)
            torch.cuda.synchronize();dense_ms=(time.perf_counter()-tick)*1000
            dense_path=output/f"{start+2:04d}.npy"
            with dense_path.open("xb") as stream:np.save(stream,dense,allow_pickle=False)
            rows.append(dict(clip_id=window[0]["clip_id"],endpoint_index=start+2,poses=poses,
                rgb=[dict(path=str(p),sha256=sha(p)) for p in paths],sparse=sparse,
                dense_path=str(dense_path),dense_sha256=sha(dense_path),
                timing=dict(sparse_ms=match_ms,dense_ms=dense_ms)))
            print(json.dumps(dict(clip_id=window[0]["clip_id"],candidates=len(sparse["points"]),
                accepted=sum(sparse["accepted"]),sparse_ms=match_ms)),flush=True)
    return dict(schema="nf-g6-predictions-v1",status="COMPLETED",capture=str(capture),
        manifest_sha256=sha(manifest_path),rows=rows,model=dict(name="Hypersim Small518/max20",
        weights_sha256=sha(weights),source=identity,backend=asdict(torch_observation(model=model))),
        model_inference_calls=12,matcher_calls=12,uses_native_depth=False,uses_ideal_metric_poses=True,
        timing={k:distribution([r["timing"][k] for r in rows]) for k in ("sparse_ms","dense_ms")})


def stats(values):
    values=np.asarray(values)
    return distribution(values[np.isfinite(values)].tolist())


def points_world(pixels, depths, pose):
    f=640/(2*np.tan(np.deg2rad(50)))
    rays=np.column_stack(((pixels[:,0]-319.5)/f,(pixels[:,1]-179.5)/f,np.ones(len(pixels))))
    rotation, translation=pose_matrix(pose)
    return (rays*depths[:,None])@rotation.T+translation


def target_mask(world, objects):
    mask=np.zeros(len(world),bool)
    for obj in objects:
        if obj["name"] not in ("bar","flat_mark","wall_mark","mark","flush_wall_mark"):
            continue
        center=np.asarray(obj["center_m"]);half=np.asarray(obj["size_m"])/2
        mask|=(np.abs(world-center)<=half+.005).all(-1)
    return mask


def evaluate(predictions, output):
    receipt=read(predictions/"result.json")
    capture=Path(receipt["capture"])
    assert sha(capture/"model/sensor_manifest.json")==receipt["manifest_sha256"]
    # Evaluator-only source access begins here, after prediction caches exist.
    spec=read(capture/"evaluator/spec.json")
    cases=spec["cases"]
    rows=[]
    for row in receipt["rows"]:
        i=row["endpoint_index"];case=cases[i];pose=row["poses"][-1]
        assert case["camera"]=={k:pose[k] for k in case["camera"]}
        for rgb in row["rgb"]:assert sha(Path(rgb["path"]))==rgb["sha256"]
        dp=Path(row["dense_path"]);assert sha(dp)==row["dense_sha256"]
        dense=np.load(dp,allow_pickle=False)
        native_path=capture/"evaluator/native"/f"{i:04d}.npy"
        native=np.load(native_path,allow_pickle=False)
        s=row["sparse"];pixels=np.asarray(s["points"],dtype=int).reshape(-1,2)
        dep=np.asarray(s["depth_m"]);accepted=np.asarray(s["accepted"],bool)
        true_depth=native[pixels[:,1],pixels[:,0]]
        native_world=points_world(pixels,true_depth,pose)
        predicted_world=points_world(pixels,dep,pose)
        # Heading-forward range (pitch compensated), with ideal floor .12 m.
        yaw=np.deg2rad(pose["yaw"]);heading=np.array([np.cos(yaw),np.sin(yaw),0])
        origin=np.array([pose[k] for k in ("x","y","z")])
        true_range=(native_world-origin)@heading
        pred_range=(predicted_world-origin)@heading
        reference_valid=(true_depth>.08)&(true_depth<100)
        target=target_mask(native_world,case["objects"])&reference_valid
        near=accepted&(pred_range<=3)&(pred_range>.08)
        true_near=reference_valid&(true_range<=3)&(true_range>.08)
        upper=points_world(pixels,np.asarray(s["interval_m"]).reshape(-1,2)[:,1],pose)
        near&=((upper-origin)@heading<=3)
        body_head=near&(predicted_world[:,2]-.12>=.65)&(predicted_world[:,2]-.12<1.85)
        interval=np.asarray(s["interval_m"]).reshape(-1,2)
        relative_inverse_width=(1/interval[:,0]-1/interval[:,1])*dep
        failures={"low_correlation":np.asarray(s["correlation"])<.75,
            "ambiguous_depth":np.asarray(s["margin"])<.05,
            "wide_interval":relative_inverse_width>.5,
            "low_parallax":np.asarray(s["parallax_px"])<1.}
        # Dense model residual at these RGB-selected points, no native selection.
        d=dense[pixels[:,1],pixels[:,0]]
        neighbours=np.stack([dense[np.clip(pixels[:,1]+dy,0,359),np.clip(pixels[:,0]+dx,0,639)]
            for dx,dy in ((-8,0),(8,0),(0,-8),(0,8))])
        residual=np.abs(1/np.maximum(d,.08)-np.median(1/np.maximum(neighbours,.08),axis=0))
        # Target coverage denominator includes native visible surface, not only proposals.
        yy,xx=np.indices(native.shape);allpixels=np.column_stack((xx.ravel(),yy.ravel()))
        full_world=points_world(allpixels,native.ravel(),pose)
        full_target=target_mask(full_world,case["objects"])&(native.ravel()>.08)
        groups={}
        for label,mask in (("target",target),("non_target",~target&reference_valid)):
            groups[label]=dict(candidates=int(mask.sum()),accepted=int((mask&accepted).sum()),
                near=int((mask&near).sum()),body_head_near=int((mask&body_head).sum()),
                unknown=int((mask&~accepted).sum()),true_near=int((mask&true_near).sum()),
                false_near=int((mask&near&~true_near).sum()),
                false_body_head_near=int((mask&body_head&~true_near).sum()),
                best_depth_m=stats(dep[mask]),margin=stats(np.asarray(s["margin"])[mask]),
                relative_inverse_width=stats(relative_inverse_width[mask]),
                overlapping_rejections={key:int((mask&value).sum()) for key,value in failures.items()},
                parallax_px=stats(np.asarray(s["parallax_px"])[mask]),
                correlation=stats(np.asarray(s["correlation"])[mask]),
                rotation_correlation=stats(np.asarray(s["rotation_correlation"])[mask]),
                dense_depth_m=stats(d[mask]),dense_inverse_residual=stats(residual[mask]),
                estimated_depth_m=stats(dep[mask&accepted]),native_forward_m=stats(true_range[mask]))
        rows.append(dict(clip_id=row["clip_id"],case_name=case["name"],
            native_target_pixels=int(full_target.sum()),groups=groups,
            native_sha256=sha(native_path),sparse_target_body_head_recovered=bool((target&body_head).any()),
            dense_target_near_pixels=int((full_target&(dense.ravel()<=3)&(dense.ravel()>.08)).sum()),
            prediction_timing=row["timing"],all_candidate_count=len(pixels)))
    return dict(schema="nf-g6-information-result-v1",status="COMPLETED",phase="EXPLORE",
        prediction_sha256=sha(predictions/"result.json"),spec_sha256=sha(capture/"evaluator/spec.json"),rows=rows,
        total_target_candidates=sum(r["groups"]["target"]["candidates"] for r in rows),
        total_target_false_near=sum(r["groups"]["target"]["false_near"] for r in rows),
        total_non_target_false_near=sum(r["groups"]["non_target"]["false_near"] for r in rows),
        timing=receipt["timing"],model_inference_calls=0,
        limitation="Ideal metric poses, posed static windows, native-bound target masks. No IMU-only, dynamic, or natural accuracy claim.",
        app_promotion=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("action",choices=("predict","evaluate"))
    p.add_argument("--capture",type=Path);p.add_argument("--predictions",type=Path)
    p.add_argument("--metric-source",type=Path);p.add_argument("--weights",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    if not output.is_relative_to((ROOT/"artifacts.local").resolve()):raise ValueError("Canonical artifacts required")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    meta=dict(code_sha256={f.name:sha(f) for f in (Path(__file__),Path(__file__).with_name("sparse_structure.py"))},
        protocol_sha256=sha(Path(__file__).with_name("STRUCTURE_INFORMATION_20260907.md")))
    (output/"started.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    try:
        result=predict(args.capture.resolve(),output,args.metric_source.resolve(),args.weights.resolve()) if args.action=="predict" else evaluate(args.predictions.resolve(),output)
        (output/"result.json").write_text(json.dumps({**meta,**result},indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k not in ("rows","model")}))
    except BaseException as exc:
        (output/"failure.json").write_text(json.dumps(dict(error=repr(exc))),encoding="utf-8")
        raise
    finally:
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=="__main__":main()
