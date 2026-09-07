"""Receipt-bound cache attribution and one fixed support challenger."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from near_field import Camera, NearFieldEncoder, HEIGHT_EDGES, DIRECTIONS, HEIGHTS, surface_support
from run_ground_anchor import read_verified, compare, ROOT, RECEIPT_SHA
from diagnose_depth_loss import sha, distribution
from probe_source import generate_cases
from research_backend import torch_observation

G1_SHA = "deba17e8b67cf411ef8caebddd4b53a22e3cfc58f1783d907e5011d658c60214"
MODES = ("depth", "elevation", "dual")


def tensors(encoder, array, plane):
    d = torch.as_tensor(array, device=encoder.device)
    x = d*encoder.ray_x
    if plane is None:
        height = encoder.camera.camera_height_m+d*encoder.ray_z
    else:
        a,b,c = plane
        height = d*encoder.ray_z-a*x-b*d*encoder.ray_y-c
    band = torch.bucketize(height, torch.tensor(HEIGHT_EDGES,device=encoder.device),right=True)-1
    valid = torch.isfinite(d)&(d>.08)&(d<12)&(encoder.ray_x>0)
    eligible = valid&(band>=0)&(band<3)
    support = surface_support(d,height,eligible)
    return d,x,height,band,valid,eligible,support


def attribution(encoder, native, predicted, plane, reference, evidence):
    _,nx,_,nb,_,_,ns = tensors(encoder,native,None)
    _,x,height,band,valid,eligible,support = tensors(encoder,predicted,plane)
    rows = []
    for direction,height_band in np.argwhere(reference & ~evidence.alerts()):
        cell = encoder.direction == int(direction)
        candidate = cell & eligible & (band==int(height_band))
        near = candidate & (x<=3)
        mask = cell & ns & (nb==int(height_band)) & (nx<=3)
        # Reference-conditioned disjoint partition, not model inputs.
        remaining = mask.clone()
        buckets = {}
        conditions = (
            ("invalid",~valid), ("forward_over_3m",x>3),
            ("height_below",height<HEIGHT_EDGES[0]),
            ("height_above",height>=HEIGHT_EDGES[-1]),
            ("changed_height_band",band!=int(height_band)),
            ("same_band_unsupported",~support),
            ("same_band_supported",support))
        for name, condition in conditions:
            selected = remaining & condition
            buckets[name] = int(selected.sum())
            remaining &= ~condition
        assert not bool(remaining.any())
        native_count = int(mask.sum())
        assert native_count == sum(buckets.values())
        finite = mask & valid
        def stats(values):
            return distribution(values.detach().cpu().numpy())
        row = dict(direction=DIRECTIONS[direction],height=HEIGHTS[height_band],
                   state=evidence.region_state[direction][height_band],
                   region_candidates=int(candidate.sum()),near_candidates=int(near.sum()),
                   near_supported=int((near&support).sum()),
                   minimum_candidate_forward_m=float(x[candidate].min()) if bool(candidate.any()) else None,
                   native_supported_near_pixels=native_count,buckets=buckets,
                   overlapping_far=int((finite&(x>3)).sum()),
                   overlapping_wrong_band=int((finite&(band!=int(height_band))).sum()),
                   predicted_forward_at_reference=stats(x[finite]),
                   predicted_height_at_reference=stats(height[finite]))
        rows.append(row)
    return rows


@torch.inference_mode()
def execute(baseline, original):
    started = time.perf_counter()
    identities = {}
    g1 = json.loads(read_verified(baseline/"result.json",G1_SHA,identities))
    cache = {Path(path).resolve(): read_verified(Path(path),expected,identities)
             for path,expected in g1["verified_inputs"].items()}
    original_receipt = original/"result.json"
    if hashlib.sha256(cache[original_receipt]).hexdigest()!=RECEIPT_SHA:
        raise ValueError("Unexpected original receipt")
    prior = json.loads(cache[original_receipt])
    source = Path(prior["replay"]["source"]["model_dir"]).resolve()
    manifest = json.loads(cache[source/"manifest.json"])
    records = prior["replay"]["rows"]
    if len(records)!=11 or len(g1["rows"])!=11:
        raise ValueError("Require eleven complete frames")
    loaded=[]
    for i,(record,frame,grow) in enumerate(zip(records,manifest["episodes"][0]["frames"],g1["rows"])):
        if any(row["sample_index"]!=i for row in (record,frame,grow)):
            raise ValueError("Frame identity/order mismatch")
        native=np.load(io.BytesIO(cache[(source/frame["depth_path"]).resolve()]),allow_pickle=False)
        predicted=np.load(io.BytesIO(cache[Path(record["predicted_path"]).resolve()]),allow_pickle=False)
        loaded.append((record,grow,native,predicted))
    load_s=time.perf_counter()-started
    if not torch.cuda.is_available():
        raise RuntimeError("Recorded CUDA backend unavailable; no silent fallback")
    encoders={Camera(**r[0]["camera"]):NearFieldEncoder(Camera(**r[0]["camera"]),device="cuda:0") for r in loaded}
    totals={m:dict(tp=0,fp=0,fn=0,tn=0,unknown=0,unknown_positive=0) for m in MODES}
    times={m:[] for m in MODES}
    rows=[]
    for record,grow,native,predicted in loaded:
        encoder=encoders[Camera(**record["camera"])]
        plane=grow["fit"]["plane"]
        if grow["fit"]["status"]!="ACCEPTED":
            raise ValueError("Unexpected failed fit in frozen baseline")
        reference=np.asarray(record["native"]["near_field"]["alerts"],bool)
        outputs={}
        for mode in MODES:
            tick=time.perf_counter()
            evidence=encoder.encode(predicted,ground_plane=plane,support_mode=mode)
            times[mode].append((time.perf_counter()-tick)*1000)
            counts=compare(evidence.alerts(),evidence.region_state,reference)
            for key,value in counts.items(): totals[mode][key]+=value
            outputs[mode]=dict(counts=counts,evidence=evidence.summary(),alerts=evidence.alerts().tolist())
            if mode=="depth":
                np.testing.assert_array_equal(evidence.alerts(),grow["arms"]["ground_only"]["alerts"])
                assert evidence.region_state==grow["arms"]["ground_only"]["states"]
                missed=attribution(encoder,native,predicted,plane,reference,evidence)
        k=grow["fit"]["scale"]
        scaled_plane=(plane[0],plane[1],k*plane[2])
        scaled=encoder.encode(predicted*k,ground_plane=scaled_plane)
        np.testing.assert_array_equal(scaled.alerts(),grow["arms"]["scale_and_ground"]["alerts"])
        rows.append(dict(sample_index=record["sample_index"],arms=outputs,
                         ground_only_misses=missed,
                         scaled_misses=attribution(encoder,native,predicted*k,scaled_plane,reference,scaled)))
    assert sum(len(r["ground_only_misses"]) for r in rows)==5
    controls=[]
    control_totals={m:dict(tp=0,fp=0,fn=0,tn=0,unknown=0,unknown_positive=0) for m in MODES}
    for case in generate_cases():
        camera=Camera(**case["camera"])
        if camera not in encoders: encoders[camera]=NearFieldEncoder(camera,device="cuda:0")
        arms={}
        for mode in MODES:
            e=encoders[camera].encode(case["depth"],ground_plane=(0.,0.,-camera.camera_height_m),support_mode=mode)
            counts=compare(e.alerts(),e.region_state,case["expected"])
            for key,value in counts.items(): control_totals[mode][key]+=value
            arms[mode]=dict(counts=counts,alerts=e.alerts().tolist())
        controls.append(dict(name=case["name"],family=case["family"],arms=arms,
                             depth_sha256=hashlib.sha256(case["depth"].tobytes()).hexdigest()))
    assert len(controls)==88
    assert control_totals["depth"]["tp"]==240 and control_totals["depth"]["fp"]==1
    return dict(status="COMPLETED",phase="EXPLORE",verified_inputs=identities,
                backend=asdict(torch_observation(output=encoder.ray_x)),
                reused_backend=g1["backend"],frames=11,model_inference_calls=0,simulator_launches=0,
                totals=totals,rows=rows,control_totals=control_totals,controls=controls,
                timing=dict(load_hash_decode_s=load_s,total_processing_s=time.perf_counter()-started,
                            arm_ms={m:distribution(t) for m,t in times.items()},
                            exclusions="Python/import startup and receipt writing; cached cost, not online latency"),
                registration="PENDING",promotion=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to((ROOT/"artifacts.local").resolve()): raise ValueError("Canonical artifacts required")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol=Path(__file__).with_name("SURFACE_SUPPORT_20260907.md")
    receipt=dict(schema="nearfield-surface-support-v1",code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
                 protocol_sha256=sha(protocol),python=sys.executable,torch=torch.__version__)
    (output/"protocol.md").write_bytes(protocol.read_bytes())
    (output/"started.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    try:
        result={**receipt,**execute(args.baseline.resolve(strict=True),args.original.resolve(strict=True))}
        (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({k:result[k] for k in ("totals","control_totals","timing")}))
        for row in result["rows"]:
            if row["ground_only_misses"]: print(json.dumps({"frame":row["sample_index"],"misses":row["ground_only_misses"]}))
    except BaseException as exc:
        (output/"failure.json").write_text(json.dumps({**receipt,"error":repr(exc)},indent=2),encoding="utf-8")
        raise
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__=="__main__": main()
