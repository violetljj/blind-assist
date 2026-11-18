"""Score frozen, pre-alarm RGB verdicts against measured range and the prior return regions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import holoset_return_time_20261118 as d8
from holoset_return_objects_20261118 import coordinates
from holoset_rgb_contradiction_20261119 import RGBVerifier, holoset_views, match_pair


ROOT = d8.WORK / "holoset-contradiction-20261119"
LABELS = d8.WORK / "holoset-return-20261118/scene/labels"


def summarize(rows):
    if not rows:
        return {"n":0}
    a = np.stack(rows)
    original, measured, mono_rgb, tri, identified, withdrawn, has_native, native = a.T
    good = measured-original > np.maximum(.20,.20*original)
    flag = withdrawn.astype(bool)
    native_ok = has_native.astype(bool)
    native_good = native-original > np.maximum(.20,.20*original)
    nt = lambda ar: int(np.count_nonzero(ar))
    both = flag & native_ok
    def median(x):
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else None
    return dict(n=len(a), reference_too_near=nt(good), rgb_matched=nt(identified),
                rgb_too_near=nt(flag), confirmed=nt(flag & good), unconfirmed=nt(flag & ~good),
                native_supported=nt(native_ok), native_too_near=nt(native_ok & native_good),
                native_flags=nt(both), native_confirmed=nt(both & native_good),
                on_flags_medians=dict(d8=median(original[flag]), measurement=median(measured[flag]),
                                      rgb_mono=median(mono_rgb[flag]), triangulated=median(tri[flag])),
                d8_depth_median=median(original), measured_depth_median=median(measured))


def point_panel(cap, ts, target_no, u, v, region, confirmed, flag):
    """Render paired measured errors and RGB-only flags in the actual previous exposure."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",18)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",15)
    canvas = Image.new("RGB",(1050,249),(252,251,248))
    draw = ImageDraw.Draw(canvas)
    with Image.open(cap.directory/"rgb"/f"{ts}.png") as im:
        canvas.paste(im.resize((295,166)),(10,49))
        crop = (370,0,760,108)
        canvas.paste(im.crop(crop).resize((720,200)),(320,40))
    draw.rectangle((10+370*295/760,49,305,49+108*166/428),outline=(254,210,63),width=2)
    draw.text((10,7),f"回程第{target_no}个报警的前一张 D8 曝光",font=font,fill=(32,41,49))
    for mask,color in ((region & confirmed & ~flag,(250,174,30)),
                       (region & confirmed & flag,(66,228,254)),(region & ~confirmed & flag,(237,84,125))):
        for x,y in zip(u[mask],v[mask]):
            px,py=320+(x-370)*720/390,40+y*200/108
            draw.ellipse((px-2.7,py-2.7,px+2.7,py+2.7),fill=color,outline=(46,50,52))
    draw.text((10,220),"原图黄框在右侧放大",font=small,fill=(55,63,68))
    return canvas


def rejection_counts(verifier,ts,dates,maps,regions):
    """Count points passing successive pair tests in two distinct causal witnesses."""
    a=verifier.views[ts]
    gates=[]
    for d in dates:
        b=verifier.views[d]
        _,m=match_pair(verifier.image(ts),verifier.image(d),a.K,b.K,np.linalg.inv(b.c2w)@a.c2w,diagnostics=True)
        u,v=[np.rint((k-.5)/2).astype(int) for k in maps]
        gates.append({k:arr[v.clip(0,arr.shape[0]-1),u.clip(0,arr.shape[1]-1)] for k,arr in m.items()})
    if not gates:
        return {}
    counts={}
    active=np.ones((len(dates),*maps[0].shape),bool)
    for k in gates[0]:
        active &= np.stack([g[k] for g in gates])
        ok=(active.sum(axis=0)>=2)&active[np.array(dates)>ts].any(axis=0)
        counts[k]={name:int((ok&regions[name]).sum()) for name in ("far_strip","pale_lip","above_lip")}
    return counts


def run(replay):
    seq = d8.SEQ
    cap = d8.original.Recording(d8.HEAD/seq)
    records = d8.original.records(cap.directory/"geometry.jsonl")
    verifier=RGBVerifier(holoset_views(cap,ROOT/"rgb"/seq))
    time_result=replay/f"result-{seq}.json"
    if not time_result.exists():
        time_result=replay/"result.json"
    clock={r["rgb_ts"]:r for r in json.loads(time_result.read_bytes())["sequences"][seq]["reports"]}
    by_time = {r["rgb_ts"]:r for r in records if r.get("rgb_ts") and r["tag"] not in ("no_depth_pose","no_rgb_match")}
    cases = [r for r in json.loads((d8.TUBE/"d8/attribution.json").read_bytes())["results"] if r["case"] in (7,8)]
    net = d8.I.D.make_head(8)
    net.load_state_dict(torch.load(d8.WORK/"ase-cane-reach-pilot-20261015/decision_head8_all.pt",
                                   map_location="cpu",weights_only=True))
    net.eval()
    result = []
    panels = []
    for case in cases:
        target_ts = case["rgb_ts"]
        current = by_time[target_ts]
        if current["depth_ts"] != case["depth_ts"]:
            raise ValueError("Ambiguous target timestamp")
        target,clean_target = d8.remove_tube(cap,current)
        parts, individual, latest_gates = {}, {}, {}
        history = {k:[] for k in ("untouched","rgb_all","rgb_regions","measured_regions","measured_all")}
        mask_file = replay/"masks"/seq/f"{target_ts}.npz"
        with np.load(mask_file) as archive:
            for key in sorted(archive.files):
                ts = int(key)
                row = by_time[ts]
                raw,clean = d8.remove_tube(cap,row)
                flag = archive[key]
                with np.load(replay/"evidence"/seq/str(target_ts)/f"{ts}.npz") as e:
                    mono_rgb,tri,lower,valid = (e[k] for k in ("mono_rgb_m","tri_m","lower_m","valid"))
                if not np.array_equal(flag,valid & (lower > mono_rgb + np.maximum(.20,.20*mono_rgb))):
                    raise ValueError("Verdicts differ from the saved independent RGB evidence")
                label = cv2.imread(str(LABELS/f"{ts}.png"),cv2.IMREAD_GRAYSCALE)
                label = cv2.remap(label,*raw["maps"],cv2.INTER_NEAREST)
                accepted = (clean["depth"]>0)&(clean["depth"]<8)
                coord = coordinates(clean["points"],target["pose"],current["floor_y"])
                grid = np.zeros_like(flag)
                grid[accepted] = coord[0]
                u,v = raw["maps"]
                corner = (u>=380)&(v<107)
                critical = corner & np.isin(label,[14,15,17])
                native = cap.registration(row,native=True)[-1]
                confirmed = (raw["ref"]>0)&(raw["ref"]<8)&(raw["ref"]-raw["depth"]>np.maximum(.20,.20*raw["depth"]))
                if key==max(archive.files):
                    panels.append(point_panel(cap,ts,case["case"]-6,u,v,critical & grid,confirmed,flag))
                for arm,mask in (("untouched",np.zeros_like(flag)),("rgb_all",flag),("rgb_regions",flag&critical),
                                 ("measured_regions",confirmed&critical),("measured_all",confirmed)):
                    history[arm].append({**clean,"depth":np.where(mask,0.,clean["depth"]),
                                         "points":clean["points"][~mask[accepted]]})
                data = np.stack([raw["depth"],raw["ref"],mono_rgb,tri,valid,flag,(native>0)&(native<8),native],axis=-1)
                regions = {"all_non_tube_grid":grid,"far_strip":grid&corner&(label==14),
                           "pale_lip":grid&corner&(label==15),"above_lip":grid&corner&(label==17),
                           "upper_right":grid&corner,"other_grid":grid&~corner}
                tube_distance = cv2.distanceTransform((label!=1).astype(np.uint8),cv2.DIST_L2,5)
                regions["far_strip_away_from_tube"] = regions["far_strip"] & (tube_distance>3)
                if key==max(archive.files):
                    diag=next(c["diagnostics"] for c in clock[target_ts]["changes"] if c["source_rgb_ts"]==ts)
                    if any(t>=clock[target_ts]["cutoff"] for t in diag["witnesses"]):
                        raise ValueError("Non-causal witness")
                    latest_gates=rejection_counts(verifier,ts,diag["witnesses"],raw["maps"],regions)
                individual[key] = {}
                for name,roi in regions.items():
                    points = data[roi]
                    individual[key][name] = summarize(list(points))
                    parts.setdefault(name,[]).extend(points)
        scores = {k:d8.score(net,d8.original.features(frames+[clean_target],current["floor_y"],d8.V))
                  for k,frames in history.items()}
        result.append(dict(case=case["case"],target=target_ts,scores=scores,latest_gates=latest_gates,
                           all_history={k:summarize(v) for k,v in parts.items()},
                           by_exposure=individual))
    out=replay/"point_audit.json"
    out.write_text(json.dumps(result,indent=1,allow_nan=False),encoding="utf-8")
    from PIL import Image, ImageDraw, ImageFont
    figure=Image.new("RGB",(1050,552),(252,251,248))
    for i,panel in enumerate(panels):
        figure.paste(panel,(0,i*249))
    draw=ImageDraw.Draw(figure)
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",17)
    draw.text((10,501),"同一射线的实测距离用于事后核对。只画目标区域中进入 D8 网格的历史点。",font=font,fill=(43,49,54))
    for x,color,text in ((12,(250,174,30),"D8 估得过近，RGB 未标出"),(310,(66,228,254),"D8 估得过近，RGB 标出"),(613,(237,84,125),"RGB 标出，实测未确认")):
        draw.ellipse((x,533,x+9,542),fill=color,outline=(55,65,68))
        draw.text((x+17,525),text,font=font,fill=(43,49,54))
    figure.save(replay/"point_review.png")
    for r in result:
        print(r["case"],[(k,v.get("confirmed",0),v.get("rgb_too_near",0),v.get("reference_too_near",0),v["n"])
                         for k,v in r["all_history"].items()])


if __name__=="__main__":
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay",type=Path)
    run(parser.parse_args().replay)
