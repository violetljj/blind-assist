"""Post-score attribution only: inspect the four weak pixels, no tuning."""
import argparse
import io
import json
from pathlib import Path
import torch
import numpy as np
from near_field import Camera,NearFieldEncoder,surface_support
from run_surface_support import tensors,G1_SHA
from run_ground_anchor import read_verified,ROOT
from diagnose_depth_loss import sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/"artifacts.local").resolve()):
        raise ValueError("Require new canonical diagnostic output")
    identities={}
    g1=json.loads(read_verified(args.baseline/"result.json",G1_SHA,identities))
    paths={Path(path).resolve():digest for path,digest in g1["verified_inputs"].items()}
    prior_path=next(path for path in paths if path.name=="result.json")
    prior=json.loads(read_verified(prior_path,paths[prior_path],identities))
    record=prior["replay"]["rows"][5]
    prediction_path=Path(record["predicted_path"]).resolve()
    depth=np.load(io.BytesIO(read_verified(prediction_path,paths[prediction_path],identities)),allow_pickle=False)
    torch.set_num_threads(1)
    encoder=NearFieldEncoder(Camera(**record["camera"]),device="cuda:0")
    with torch.inference_mode():
        d,x,h,band,valid,eligible,support=tensors(encoder,depth,g1["rows"][5]["fit"]["plane"])
        near=eligible&(band==0)&(x<=3)&(encoder.direction==0)
        # Constant depth makes compatibility universally true. The unchanged
        # three-adjacent-eligible-pixel requirement remains the only restriction.
        geometry_only=surface_support(torch.ones_like(d),h,eligible,"depth")
        pixels=[]
        for v,u in torch.nonzero(near).cpu().tolist():
            pixels.append(dict(v=v,u=u,forward_m=float(x[v,u]),height_m=float(h[v,u]),
                               any_eligible_triple=bool(geometry_only[v,u]),
                               depth_supported=bool(support[v,u])))
        result=dict(schema="nearfield-weak-support-attribution-v1",sample_index=5,
                    verified_inputs=identities,pixels=pixels,
                    eligible_triple_near_count=int((geometry_only&near).sum()),
                    code_sha256=sha(Path(__file__)),
                    interpretation="Removing consistency entirely cannot support these near candidates if eligible_triple_near_count is zero.",
                    scoring_or_algorithm_change=False)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__": main()
