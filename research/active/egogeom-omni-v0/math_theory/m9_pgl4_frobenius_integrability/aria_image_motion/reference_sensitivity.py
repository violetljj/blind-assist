"""Read the same frozen out-of-recording scores against alternate reference definitions."""
import json
from pathlib import Path
import numpy as np
from build import OUT, dump, sha
from rank import BW_SPEEDS_PROXY, make_episodes, reference, counts

REF=OUT.parent/"aria_reference_heading"
MODES=("centered_full","strict_full","strict_past_points_full_ground","strict_past")


def main():
    old=json.loads((OUT/"results.json").read_text())
    summary={}
    files=sorted(REF.glob("aria_walk_*.npz"))
    assert len(files)==19
    for ver in ("v4","v41"):
        report={}
        for arm in ("geometry","aligned"):
            scores,dist,alert={},{},{}
            folds=json.loads((OUT/ver/"folds.json").read_text())["splits"]
            hashes={n:h for f in folds for n,h in f["arms"][arm]["predictions"].items()}
            for p in files:
                src=OUT/ver/arm/p.name
                assert sha(src)==hashes[p.stem]
                with np.load(src) as f:
                    scores[p.stem],dist[p.stem],alert[p.stem]=f["score"],f["distance"],f["alert"]
            modes={}
            for mode in MODES:
                clips=[]
                for p in files:
                    n=p.stem
                    c=reference(n)
                    with np.load(p) as f:
                        np.testing.assert_allclose(c.episodes[0]["rows"],f["frozen"],equal_nan=True,atol=1.e-12)
                        r=f[mode]
                    c.episodes=make_episodes(n,r,BW_SPEEDS_PROXY,kind="blind")
                    valid=c.episodes[0]["valid"]
                    c.clear=valid&(r[:,0]>4.)&(r[:,2]<=.5)
                    c.position=np.cumsum(np.where(valid,r[:,3],0.),dtype=float)
                    c.known=np.isfinite(dist[n])
                    clips.append(c)
                res=counts(clips,scores,dist,alert)
                pooled=res["operating"]["pooled"]
                modes[mode]=dict(hazard_frames=res["evaluated_hazard_frames"],clear_frames=pooled["known_clear_frames"],
                    true_opportunities=sum(r["hazard_window"] for r in res["events"]),
                    avoidable=pooled["avoidable"],timely=pooled["timely"],range_hits=res["correct_distance"],
                    filled_mae=res["distance_mae_fill6"],auroc=res["auroc_hazard_clear"],
                    posthoc_timely=res["pooled_posthoc"]["timely"],posthoc_cut=res["pooled_posthoc_cut"],
                    posthoc_repeat_per_km=res["pooled_posthoc"]["clear_warnings_per_km"])
                if mode=="centered_full":
                    assert modes[mode]["range_hits"]==old["versions"][ver][arm]["correct_distance"]
                    assert modes[mode]["posthoc_timely"]==old["versions"][ver][arm]["pooled_posthoc"]["timely"]
            report[arm]=modes
        summary[ver]=report
    dump(OUT/"reference_sensitivity.json",dict(versions=summary,original=sha(OUT/"results.json"),
         refs={p.name:sha(p) for p in files},code=sha(Path(__file__))))
    for ver,arms in summary.items():
        print(ver,[(k,v["range_hits"],v["true_opportunities"],v["posthoc_timely"],round(v["auroc"],3))
                   for k,v in arms["aligned"].items()])

if __name__=="__main__":
    main()
