from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import read_json, sha256, write_json


def _rows(value: dict) -> list[dict]:
    rows=value.get("sources",value.get("reviews"))
    if not isinstance(rows,list): raise ValueError("review rows missing")
    return rows


def finalize(root: Path) -> dict:
    inputs=read_json(root/"review_inputs.json"); a=read_json(root/"reviewer_a_raw.json"); b=read_json(root/"reviewer_b_raw.json")
    expected={row["source_id"]:row["sha256"] for row in inputs["inputs"]}
    if a.get("independent_review") is not True or a.get("candidate_alerts_inspected") is not False: raise ValueError("review A isolation failed")
    constraints=b.get("independence_constraints",{})
    if constraints.get("other_reviewer_outputs_viewed") is not False or constraints.get("candidate_alerts_viewed") is not False: raise ValueError("review B isolation failed")
    reviews=[]
    for name,value in (("a",a),("b",b)):
        rows=_rows(value)
        actual={row["source_id"]:row.get("sha256") for row in rows}
        if actual!=expected: raise ValueError(f"review {name} input binding mismatch")
        reviews.append({"reviewer":name,"raw_sha256":sha256(root/f"reviewer_{name}_raw.json"),"prompt_sha256":sha256(root/f"reviewer_{name}_prompt.txt")})
    by_source=[]
    ar={row["source_id"]:row for row in _rows(a)}; br={row["source_id"]:row for row in _rows(b)}
    for source_id in expected:
        acceptable={False,"abstain"}
        if ar[source_id]["route_valid"] not in acceptable or br[source_id]["route_valid"] not in acceptable: raise ValueError("unexpected route admission disagreement")
        by_source.append({"source_id":source_id,"route_event_admitted":False,"disposition":"reject_or_abstain","reason":"no independently reviewable body-bound route and causal event lifecycle"})
    report={"schema":"blindassist_ustrf_sensor_replay_review_consensus_v1","workflow_id":"ustrf_event_review_v1","input_manifest_sha256":sha256(root/"review_inputs.json"),"reviews":reviews,"sources":by_source,"consensus":"fail_closed","third_model_adjudication_required":False,"event_truth_authority":False,"production_authority":False}
    write_json(root/"review_consensus.json",report); return report


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True); args=parser.parse_args()
    try: report=finalize(args.root.resolve()); print(json.dumps({"consensus":report["consensus"],"sources":len(report["sources"])})); return 0
    except (OSError,ValueError,KeyError) as error: print(json.dumps({"ok":False,"error":str(error)})); return 2


if __name__=="__main__": raise SystemExit(main())
