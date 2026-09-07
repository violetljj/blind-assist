"""Execute fusion on frozen branch caches and analytic regression controls."""
import argparse
import json
from pathlib import Path
import time
from dataclasses import asdict
import numpy as np
import torch
from evidence_fusion import BranchEvidence,fuse
from near_field import Camera,NearFieldEncoder
from run_ground_anchor import ROOT,read_verified,compare,RECEIPT_SHA
from run_surface_support import G1_SHA
from diagnose_suspended_bar import EVALUATION_SHA
from diagnose_depth_loss import sha
from probe_source import generate_cases
from research_backend import torch_observation


def scores(alerts,states,reference):
    full=compare(alerts,states,reference)
    high=compare(alerts[:,1:],np.asarray(states)[:,1:],reference[:,1:])
    a,n=alerts.any(axis=1),reference.any(axis=1)
    return dict(legacy_cells=full,body_head_diagnostic=high,
                directions=dict(tp=int((a&n).sum()),fp=int((a&~n).sum()),fn=int((~a&n).sum()),tn=int((~a&~n).sum())))


def run_rows(source_rows):
    rows=[];totals={}
    provenance={k:0 for k in ("RAW_ONLY","GROUND_ONLY","BOTH","NONE")}
    for name,raw,ground,reference in source_rows:
        merged={m:fuse(raw,ground,mode=m) for m in ("union","conditional")}
        unknown=np.full((3,3),"UNKNOWN")
        options={"raw":(raw.alerts,raw.states),"ground":(ground.alerts,ground.states) if ground is not None else (np.zeros((3,3),bool),unknown)}
        options.update({m:(np.asarray(r["diagnostic_alerts"]),r["diagnostic_states"]) for m,r in merged.items()})
        row=dict(name=name,arms={},fusion=merged)
        for mode,(alerts,states) in options.items():
            result=scores(alerts,states,reference)
            if mode not in totals:
                totals[mode]={group:{k:0 for k in values} for group,values in result.items()}
            for group,values in result.items():
                for key,value in values.items(): totals[mode][group][key]+=value
            row["arms"][mode]=result
        for key in provenance: provenance[key]+=int((np.asarray(merged["union"]["provenance"])==key).sum())
        rows.append(row)
    return dict(rows=rows,totals=totals,provenance_cells=provenance,
                union_conditional_alert_differences=sum(int((np.asarray(r["fusion"]["union"]["diagnostic_alerts"])!=np.asarray(r["fusion"]["conditional"]["diagnostic_alerts"])).sum()) for r in rows),
                union_near_directions_unknown_height=sum(d["range_state"]=="NEAR" and d["height_status"]=="UNKNOWN" for r in rows for d in r["fusion"]["union"]["directions"]))


def execute(g3,g1,original):
    tick=time.perf_counter();identities={}
    new=json.loads(read_verified(g3,EVALUATION_SHA,identities))
    old=json.loads(read_verified(g1,G1_SHA,identities))
    origin=json.loads(read_verified(original,RECEIPT_SHA,identities))
    def g3_rows():
        for row in new["rows"]:
            raw=row["arms"]["raw"];ground=row["arms"]["ground_only"]
            yield row["case_name"],BranchEvidence(raw["alerts"],raw["states"]),BranchEvidence(ground["alerts"],ground["states"]) if row["fit"]["status"]=="ACCEPTED" else None,np.asarray(row["native_alerts"],bool)
    def g1_rows():
        for row,ref in zip(old["rows"],origin["replay"]["rows"]):
            assert row["sample_index"]==ref["sample_index"]
            raw=row["arms"]["raw"];ground=row["arms"]["ground_only"]
            yield str(row["sample_index"]),BranchEvidence(raw["alerts"],raw["states"]),BranchEvidence(ground["alerts"],ground["states"]) if row["fit"]["status"]=="ACCEPTED" else None,np.asarray(ref["native"]["near_field"]["alerts"],bool)
    recent=run_rows(g3_rows());previous=run_rows(g1_rows())
    assert len(recent["rows"])==18 and len(previous["rows"])==11
    cache_seconds=time.perf_counter()-tick
    encoders={}
    def controls():
        for case in generate_cases():
            camera=Camera(**case["camera"])
            if camera not in encoders: encoders[camera]=NearFieldEncoder(camera,device="cuda:0")
            encoder=encoders[camera]
            raw=BranchEvidence.from_encoder(encoder.encode(case["depth"]))
            ground=BranchEvidence.from_encoder(encoder.encode(case["depth"],ground_plane=(0,0,-camera.camera_height_m)))
            yield case["name"],raw,ground,case["expected"]
    analytic=run_rows(controls())
    assert len(analytic["rows"])==88
    for result in (recent,previous,analytic):
        for row in result["rows"]:
            union=row["arms"]["union"]["legacy_cells"]
            assert union["tp"]>=max(row["arms"][m]["legacy_cells"]["tp"] for m in ("raw","ground"))
    return dict(status="COMPLETED",phase="EXPLORE",g3=recent,g1=previous,analytic=analytic,verified_inputs=identities,
                timing=dict(cached_branch_fusion_s=cache_seconds,total_with_analytic_s=time.perf_counter()-tick),
                model_inference_calls=0,simulator_launches=0,backend=dict(fusion="CPU TASK_NOT_GPU_SUITABLE: 9-cell boolean and metadata merge",
                analytic=asdict(torch_observation(output=next(iter(encoders.values())).ray_x))),
                scope="Retrospective branch-height cell union and separate directional output; not natural or verified-height accuracy",
                registration="PENDING",app_promotion=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--g3",type=Path,required=True);p.add_argument("--g1",type=Path,required=True)
    p.add_argument("--original",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    if not output.is_relative_to((ROOT/"artifacts.local").resolve()): raise ValueError("Canonical artifacts required")
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=Path(__file__).with_name("EVIDENCE_FUSION_20260907.md")
    meta=dict(schema="nearfield-evidence-fusion-v1",code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},protocol_sha256=sha(protocol))
    (output/"protocol.md").write_bytes(protocol.read_bytes())
    (output/"started.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    try:
        result={**meta,**execute(args.g3,args.g1,args.original)}
        (output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({k:result[k]["totals"] for k in ("g3","g1","analytic")}))
        print(json.dumps(result["timing"]))
    except BaseException as exc:
        (output/"failure.json").write_text(json.dumps({**meta,"error":repr(exc)},indent=2),encoding="utf-8")
        raise
    finally:
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=="__main__":main()
