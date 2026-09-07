"""Separate one-time prediction from repeatable cache-only evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import io

import cv2
import numpy as np
import torch

from near_field import Camera,NearFieldEncoder
from ground_anchor import fit_ground,arm_inputs
from diagnose_depth_loss import sha,distribution
from run_ground_anchor import compare,read_verified,ROOT
from rgb_replay import _decode,_metric_class
from research_backend import torch_observation


def predict(capture,output,metric_source,weights,*,dataset="hypersim",input_size=518):
    started=time.perf_counter()
    model_dir=capture/"model"
    manifest_path=model_dir/"sensor_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt=json.loads((capture/"receipt.json").read_text(encoding="utf-8"))
    if receipt["status"]!="PASS": raise ValueError("Capture must pass")
    if len(manifest["frames"])!=18: raise ValueError("Require all18 fixed views")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required by unchanged model")
    if dataset not in ("hypersim","vkitti") or input_size not in (518,1036):
        raise ValueError("Use an explicit supported metric checkpoint/input contract")
    if weights.name!=f"depth_anything_v2_metric_{dataset}_vits.pth":
        raise ValueError("Checkpoint name and declared model domain disagree")
    max_depth=20. if dataset=="hypersim" else 80.
    cls,identity=_metric_class(metric_source)
    model=cls(encoder="vits",features=64,out_channels=[48,96,192,384],max_depth=max_depth)
    weight_hash=sha(weights)
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    model.to("cuda:0").eval()
    rows=[]
    with torch.inference_mode():
        for index,frame in enumerate(manifest["frames"]):
            if frame["sample_index"]!=index: raise ValueError("Frame order changed")
            sample=_decode(model_dir,frame,manifest["calibration"])
            torch.cuda.synchronize()
            tick=time.perf_counter()
            predicted=np.asarray(model.infer_image(sample["bgr"],input_size),dtype=np.float32)
            torch.cuda.synchronize()
            ms=(time.perf_counter()-tick)*1000
            path=output/f"{index:04d}.npy"
            with path.open("xb") as f: np.save(f,predicted,allow_pickle=False)
            rows.append(dict(sample_index=index,case_name=frame["case_name"],camera=asdict(sample["camera"]),
                             rgb_path=str((model_dir/frame["rgb_path"]).resolve()),rgb_sha256=sample["rgb_sha256"],
                             native_path=str((model_dir/frame["depth_path"]).resolve()),native_sha256=sample["native_sha256"],
                             predicted_path=str(path),predicted_sha256=sha(path),inference_ms=ms))
    result=dict(schema="nearfield-distinct-predictions-v1",status="COMPLETED",rows=rows,
                capture=str(capture),manifest_sha256=sha(manifest_path),capture_receipt_sha256=sha(capture/"receipt.json"),
                model=dict(name=f"Depth Anything V2 metric {dataset} Small",source=identity,weights_sha256=weight_hash,
                           input_size=input_size,max_depth_m=max_depth,backend=asdict(torch_observation(model=model))),
                inference_calls=18,no_warmup_or_retry=True,timing=dict(inference_ms=distribution([r["inference_ms"] for r in rows]),
                total_s=time.perf_counter()-started,includes_first_call_startup=True))
    del model
    return result


def evaluate(predictions,output):
    started=time.perf_counter()
    receipt_path=predictions/"result.json"
    prior=json.loads(receipt_path.read_text(encoding="utf-8"))
    identities={str(receipt_path):sha(receipt_path)}
    capture=Path(prior["capture"])
    read_verified(capture/"receipt.json",prior["capture_receipt_sha256"],identities)
    read_verified(capture/"model/sensor_manifest.json",prior["manifest_sha256"],identities)
    # Evaluator-only case metadata; never passed to fitting or model inference.
    spec=json.loads((capture/"evaluator/spec.json").read_text(encoding="utf-8"))
    modes=("raw","ground_only","scale_and_ground")
    totals={mode:dict(tp=0,fp=0,fn=0,tn=0,unknown=0,unknown_positive=0) for mode in modes}
    groups={}
    rows=[]
    encoders={}
    fit_ms=[]
    post_ms={m:[] for m in modes}
    for record,case in zip(prior["rows"],spec["cases"]):
        if record["case_name"]!=case["name"]: raise ValueError("Case identity mismatch")
        predicted=np.load(io.BytesIO(read_verified(Path(record["predicted_path"]),record["predicted_sha256"],identities)),allow_pickle=False)
        native=np.load(io.BytesIO(read_verified(Path(record["native_path"]),record["native_sha256"],identities)),allow_pickle=False)
        read_verified(Path(record["rgb_path"]),record["rgb_sha256"],identities)
        camera=Camera(**record["camera"])
        if camera not in encoders: encoders[camera]=NearFieldEncoder(camera,device="cuda:0")
        encoder=encoders[camera]
        reference=encoder.encode(native)
        tick=time.perf_counter()
        fit=fit_ground(predicted,camera)
        fit_ms.append((time.perf_counter()-tick)*1000)
        row=dict(case_name=case["name"],family=case["family"],condition=case["condition"],
                 pair=case.get("pair"),fit=fit.summary(),native=reference.summary(),native_alerts=reference.alerts().tolist(),arms={})
        for mode in modes:
            tick=time.perf_counter()
            inputs=arm_inputs(predicted,fit,mode)
            if inputs is None:
                evidence=None; alerts=np.zeros((3,3),bool); states=[["UNKNOWN"]*3 for _ in range(3)]
            else:
                evidence=encoder.encode(inputs[0],ground_plane=inputs[1]); alerts=evidence.alerts(); states=evidence.region_state
            post_ms[mode].append((time.perf_counter()-tick)*1000)
            counts=compare(alerts,states,reference.alerts())
            for key,value in counts.items(): totals[mode][key]+=value
            for group in ("family:"+case["family"],"condition:"+case["condition"]):
                if group not in groups: groups[group]={m:dict(tp=0,fp=0,fn=0,tn=0,unknown=0,unknown_positive=0) for m in modes}
                for key,value in counts.items(): groups[group][mode][key]+=value
            row["arms"][mode]=dict(counts=counts,alerts=alerts.tolist(),states=states,evidence=evidence.summary() if evidence else None)
        rows.append(row)
    if len(rows)!=18: raise ValueError("Require eighteen cases")
    return dict(schema="nearfield-distinct-evaluation-v1",status="COMPLETED",phase="EXPLORE",rows=rows,totals=totals,groups=groups,
                fitting_failures=sum(r["fit"]["status"]!="ACCEPTED" for r in rows),verified_inputs=identities,
                prediction_receipt_sha256=sha(receipt_path),backend=asdict(torch_observation(output=encoder.ray_x)),
                timing=dict(total_processing_s=time.perf_counter()-started,fit_ms=distribution(fit_ms),
                            arm_ms={m:distribution(v) for m,v in post_ms.items()}),
                reference="native direction-height cell agreement; not target-instance or natural accuracy",
                promotion=False,registration="PENDING")


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("action",choices=("predict","evaluate"))
    p.add_argument("--capture",type=Path)
    p.add_argument("--predictions",type=Path)
    p.add_argument("--metric-source",type=Path)
    p.add_argument("--weights",type=Path)
    p.add_argument("--dataset",choices=("hypersim","vkitti"),default="hypersim")
    p.add_argument("--input-size",type=int,choices=(518,1036),default=518)
    p.add_argument("--protocol",type=Path,help="Experiment brief; defaults to the original NF-G3 brief")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to((ROOT/"artifacts.local").resolve()): raise ValueError("Canonical artifacts required")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=args.protocol or Path(__file__).with_name("DISTINCT_VIEWS_20260907.md")
    metadata=dict(code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
                  protocol_sha256=sha(protocol),python=sys.executable,torch=torch.__version__)
    (output/"protocol.md").write_bytes(protocol.read_bytes())
    (output/"started.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    try:
        result=predict(args.capture.resolve(),output,args.metric_source.resolve(),args.weights.resolve(),dataset=args.dataset,input_size=args.input_size) if args.action=="predict" else evaluate(args.predictions.resolve(),output)
        result={**metadata,**result}
        (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({k:result[k] for k in ("status","totals","groups","fitting_failures","timing") if k in result}))
    except BaseException as exc:
        (output/"failure.json").write_text(json.dumps({**metadata,"error":repr(exc)},indent=2),encoding="utf-8")
        raise
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__=="__main__": main()
