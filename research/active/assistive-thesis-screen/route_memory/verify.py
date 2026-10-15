"""Recount saved per-query decisions and check causal history/source binding."""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

sys.dont_write_bytecode = True
from pilot import sequence_cost


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--payload",type=Path,required=True)
    root=ap.parse_args().payload
    result=json.loads((root/"run02/result.json").read_text(encoding="utf-8"))
    rows=list(csv.DictReader((root/"run02/predictions.csv").open(encoding="utf-8")))
    checks={}
    for arm in result["arms"]:
        cal=[r for r in rows if r["arm"]==arm["arm"] and r["split"]=="calibration_known" and r["known_match_correct"]=="True"]
        expected=float(np.quantile([float(r["confidence_margin"]) for r in cal],.05))
        assert expected==arm["confidence_threshold"]
        checks[arm["arm"]+"_calibration_only_threshold"]=True
        known=[r for r in rows if r["arm"]==arm["arm"] and r["split"]=="evaluation_known"]
        absent=[r for r in rows if r["arm"]==arm["arm"] and r["split"]=="evaluation_map_absent"]
        count={"n":len(known),"accepted_correct":sum(r["accepted"]=="True" and r["known_match_correct"]=="True" for r in known),
               "accepted_wrong":sum(r["accepted"]=="True" and r["known_match_correct"]=="False" for r in known),
               "abstained":sum(r["accepted"]=="False" for r in known)}
        assert all(count[k]==arm["known"][k] for k in count)
        assert sum(r["accepted"]=="True" for r in absent)==arm["map_absent"]["false_accepts"]
        checks[arm["arm"]+"_recount"]=True
    d=np.load(root/"run02/descriptors_and_distance.npz")["distance"]
    for length in [5,10]:
        all_cost=sequence_cost(d,length)
        prefix_cost=sequence_cost(d[:101],length)
        assert np.array_equal(all_cost[:101],prefix_cost)
        checks[f"causal_prefix_invariance_length_{length}"]=True
    manifest=[json.loads(l) for l in (root/"run02/sample_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    ref=[m for m in manifest if m["role"]=="summer_reference"]
    qry=[m for m in manifest if m["role"]=="winter_query"]
    assert len(ref)==len(qry)==358
    assert not set(m["sha256"] for m in ref)&set(m["sha256"] for m in qry)
    assert all(a["upstream_frame_number"]==b["upstream_frame_number"]==10+i*100 for i,(a,b) in enumerate(zip(ref,qry)))
    checks["358_independent_season_pairs_matching_upstream_sampling"]=True
    ablation=json.loads((root/"run03/result.json").read_text(encoding="utf-8"))
    arows=list(csv.DictReader((root/"run03/predictions.csv").open(encoding="utf-8")))
    acal=[r for r in arows if r["arm"]=="absolute_only" and r["split"]=="calibration_known" and r["known_match_correct"]=="True"]
    assert float(np.quantile([float(r["absolute_residual"]) for r in acal],.95))==ablation["maximum_absolute_residual"]
    for arm, splits in ablation["metrics"].items():
        for split, metrics in splits.items():
            subset=[r for r in arows if r["arm"]==arm and r["split"]==split]
            assert len(subset)==metrics["n"]
            accepted=[r for r in subset if r["accepted"]=="True"]
            if split=="evaluation_map_absent":assert len(accepted)==metrics["false_accepts"]
            else:
                assert sum(r["known_match_correct"]=="True" for r in accepted)==metrics["accepted_correct"]
                assert sum(r["known_match_correct"]=="False" for r in accepted)==metrics["accepted_wrong"]
    checks["posthoc_ablation_recount_and_calibration_only_threshold"]=True
    with zipfile.ZipFile(root/"samples.zip") as z:
        truth=loadmat(io.BytesIO(z.read("samples/nordland_trimmed/sample_ground_truth.mat")))["x"]
        assert np.array_equal(truth,np.eye(358))
        checks["official_mat_identity_alignment_358"]=True
        contact=Image.new("RGB",(512,4*160),(245,245,245));draw=ImageDraw.Draw(contact)
        for row,idx in enumerate([30,120,220,300]):
            for col,entries in enumerate([ref,qry]):
                im=Image.open(io.BytesIO(z.read(entries[idx]["archive_member"]))).convert("RGB")
                im=im.resize((256,128),Image.Resampling.NEAREST)
                contact.paste(im,(col*256,row*160+24))
                draw.text((col*256+4,row*160+4),f'{entries[idx]["role"]} q={idx}',fill=(20,20,20))
        contact.save(root/"cross_season_contact_sheet.png")
    verification={"status":"PASS","checks":checks,"manifest_entries":len(manifest),
                  "predictions_recounted":len(rows),"query_history":{"length5":"current + preceding 4 sampled observations",
                  "length10":"current + preceding 9 sampled observations","known_first_query":90,"known_first_history_start":81,
                  "absent_first_query":280,"absent_first_history_start":271,"last_mapped_reference":239,
                  "future_query_reads":0,"seconds_per_sample":None},
                  "run02_sha256":hashlib.sha256((root/"run02/result.json").read_bytes()).hexdigest()}
    (root/"verification.json").write_text(json.dumps(verification,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(verification))


if __name__=="__main__":main()
