"""Attribute the post-tube D8 scores to the exact original source rays and individual exposures."""
from pathlib import Path
import json

import cv2
import numpy as np
import torch

import real_wall_holoset_score_20261118 as original
import real_wall_holoset_attribution_20261118 as replay
from spires_insert_dev_20261117 import A, I, V


WORK = Path("artifacts.local/work")
OLD = WORK / "real-wall-search-20261118"
SEQ = "research-center-seq4"
DATA = OLD / "head" / SEQ
CACHE = OLD / "holoset-d8/remote/out" / SEQ
TUBE = WORK / "holoset-tube-20261118"
ROOT = WORK / "holoset-return-20261118/spatial"
N = V.NH * V.NF


def events(frame, cur, floor, current):
    """Expand occupancy endpoints and finite visibility probes with their exact source pixel."""
    depth = frame["depth"]
    points = frame["points"]
    pose = cur["pose"]
    origin = pose[:3, 3]
    up = np.array([0., 0., 1.])
    forward = pose[:3, 2].copy()
    forward[2] = 0.
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    delta = points - origin
    x, dist, height = delta @ right, delta @ forward, points[:, 2] - floor
    grid = (abs(x) <= V.X_HALF) & (dist >= 0) & (dist < V.NF*V.CELL) & (height >= 0) & (height < V.NH*V.CELL)
    v, u = np.nonzero((depth > 0) & (depth < A.MAX_MM * .001))
    output_cell = (height[grid] // V.CELL).astype(int) * V.NF + (dist[grid] // V.CELL).astype(int)
    pixels = [v[grid] * depth.shape[1] + u[grid]]
    arm = int(not current)
    channels = [arm * N + output_cell]

    h = (np.arange(V.NH) + .5) * V.CELL
    f = (np.arange(V.NF) + .5) * V.CELL
    lateral, ahead, above = np.meshgrid([-.2, 0., .2], f, h, indexing="ij")
    world = (origin*(1-up) + lateral[..., None]*right + ahead[..., None]*forward + (floor+above)[..., None]*up).reshape(-1,3)
    q = (world - frame["pose"][:3,3]) @ frame["pose"][:3,:3]
    axial = q[:, 2]
    proj = original.FOCAL * q[:,:2] / np.where(axial > .05, axial, 1.)[:,None]
    uu,vv = ((proj + frame["center"]) / original.STEP).T
    inside = (axial>.05) & (uu>=0) & (vv>=0) & (uu<depth.shape[1]) & (vv<depth.shape[0])
    obs = np.zeros(len(q))
    obs[inside] = depth[vv[inside].astype(int),uu[inside].astype(int)]
    visible = inside & (obs > 0)
    output_cells = np.tile(np.arange(N).reshape(V.NH,V.NF).T.ravel(),3)
    for branch, subset in [(2,visible & (axial<obs-.1)), (4,visible)]:
        pixels.append(vv[subset].astype(int) * depth.shape[1] + uu[subset].astype(int))
        channels.append((branch+arm)*N+output_cells[subset])
    src = np.concatenate(pixels)
    px = np.rint(frame["maps"][0].ravel()[src]).astype(int)
    py = np.rint(frame["maps"][1].ravel()[src]).astype(int)
    # Nearest-pixel border used by remap on the original masks: out-of-image is unlabelled.
    tube = frame["tube"].ravel()[src]
    def body_mask(lateral, ahead, up_m):
        return (abs(lateral)<=.25) & (ahead>=.30) & (ahead<=2.) & (up_m>=.60) & (up_m<=1.90)
    measured = frame["measured"][grid]
    d = measured - origin
    lx, fwd, ht = d @ right, d @ forward, measured[:,2] - floor
    extra = len(src) - len(measured)
    return dict(px=px, py=py, cell=np.concatenate(channels), tube=tube, pixel=src,
                pred=depth.ravel()[src], ref=frame["ref"].ravel()[src],
                predicted_body=np.r_[body_mask(x[grid],dist[grid],height[grid]),np.zeros(extra,bool)],
                measured_body=np.r_[body_mask(lx,fwd,ht),np.zeros(extra,bool)],
                ref_forward_m=np.r_[fwd,np.full(extra,np.nan)],
                ref_lateral_m=np.r_[lx,np.full(extra,np.nan)],
                ref_height_m=np.r_[ht,np.full(extra,np.nan)])


def counts(cell):
    return np.bincount(cell, minlength=6*N).reshape(6,V.NH,V.NF)


def features(tallies, base):
    x = base.copy()
    x[[0,1,3,5]] = np.log1p(tallies[[0,1,3,5]].astype(np.float64))
    x[[2,4]] = tallies[[2,4]] / 3
    return x


def score(net, x):
    with torch.inference_mode():
        return torch.sigmoid(net(V.pad(torch.from_numpy(x[None]).float()))).item()


def upper_region_checks(frames, raw, paired_scores):
    """Follow up the 4x4 screen with the union of its two upper-right cells.

    Neighbors and per-exposure matched controls retain the entire tube mask.
    The retained-only arm sets all other measurement channels to unknown.
    """
    totals = {}
    per_frame = []
    for f, q in zip(frames,raw):
        px,py = q["px"],q["py"]
        use = ~q["tube"]
        roi = (px >= 380) & (py < 107)
        mx,my = np.rint(f["maps"][0]),np.rint(f["maps"][1])
        ok = (f["ref"] > 0) & ~f["tube"]
        n = int((ok & (mx >= 380) & (my < 107)).sum())
        pool = np.flatnonzero(ok & (mx >= 380) & (my >= 107))
        if len(pool) < n:
            raise ValueError("Too few same-half control rays")
        grid = np.zeros(f["depth"].size,bool)
        if n:
            grid[pool[np.floor((np.arange(n)+.5)*len(pool)/n).astype(int)]] = True
        masks = {
            "upper_right_half": roi, "upper_left_half": (px < 380) & (py < 107),
            "row_below_right_half": (px >= 380) & (py >= 107) & (py < 214),
            "matched_valid_right_below": grid[q["pixel"]],
        }
        for name, mask in masks.items():
            totals.setdefault(name,[]).append(counts(q["cell"][use & mask]))
        e = use & roi
        occ = e & (q["cell"]//N < 2)
        vv = dict(rt=f["rt"], valid_roi=n, valid_control=int(grid.sum()),
                  occ=int(occ.sum()), seen=int((e & (q["cell"]//N >= 4)).sum()),
                  free=int((e & ((q["cell"]//N == 2) | (q["cell"]//N == 3))).sum()))
        if occ.any():
            vv.update(median_pred_z=float(np.median(q["pred"][occ])),median_ref_z=float(np.median(q["ref"][occ])),
                      median_height_grid_m=float(np.median(((q["cell"][occ]%N)//V.NF+.5)*V.CELL)),
                      median_forward_grid_m=float(np.median(((q["cell"][occ]%V.NF)+.5)*V.CELL)))
        per_frame.append(vv)
    result={}
    for name, per in totals.items():
        group = sum(per)
        result[name] = dict(both=paired_scores(group))
    per=totals["upper_right_half"]
    for name, group in [("current",per[-1]),("history",sum(per[:-1])),("all",sum(per))]:
        arms={}
        for branch,channels in [("all",list(range(6))),("occupancy",[0,1]),("visibility_free",[2,3,4,5])]:
            partial = np.zeros_like(group)
            partial[channels] = group[channels]
            arms[branch] = paired_scores(partial)
        result["upper_right_half"][name] = arms
    result["upper_right_half"]["frame_counts"] = per_frame
    return result


def draw(frames, event_rows, case):
    gallery = []
    for frame, values in zip(frames,event_rows):
        im = cv2.imread(str(DATA/"rgb"/f"{frame['rt']}.png"))
        overlay = im.copy()
        colors = [(0,70,255),(255,210,60)]
        for j, keep in enumerate([values["cell"]//N >= 2,values["cell"]//N < 2]):
            keep &= ~values["tube"]
            for x,y in set(zip(values["px"][keep].tolist(), values["py"][keep].tolist())):
                cv2.circle(overlay,(x,y),2,colors[1-j],-1)
        im=cv2.addWeighted(overlay,.65,im,.35,0)
        for i in range(1,4):
            cv2.line(im,(i*190,0),(i*190,428),(128,128,128),1)
            cv2.line(im,(0,i*107),(760,i*107),(128,128,128),1)
        tag=f"{(frame['ts']-frames[-1]['ts'])/1e7:+.2f}s"
        cv2.putText(im,tag,(12,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.imwrite(str(ROOT/f"src-case{case}-{frame['rt']}.jpg"),im)
        gallery.append(cv2.resize(im,(380,214)))
    if len(gallery)%2: gallery.append(np.zeros_like(gallery[-1]))
    cv2.imwrite(str(ROOT/f"case{case}-source.jpg"),np.vstack([np.hstack(gallery[i:i+2]) for i in range(0,len(gallery),2)]))


def focused_scene(frames, raw, total, paired_scores):
    """Cross the chosen right-upper image sector with independently traced scene classes."""
    scene_dir = ROOT.parent / "scene"
    manifest = json.loads((scene_dir/"manifest.json").read_bytes())
    names = manifest["classes"]
    frame_blocks = []
    detail = []
    edge_distances = {}
    for i,(f,q) in enumerate(zip(frames,raw)):
        im = cv2.imread(str(scene_dir/"labels"/f"{f['rt']}.png"),cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (428,760):
            raise ValueError("Missing native semantic region map")
        region = (q["px"] >= 380) & (q["px"] < 760) & (q["py"] >= 0) & (q["py"] < 107) & ~q["tube"]
        ids = im[q["py"].clip(0,427),q["px"].clip(0,759)]
        outer = cv2.imread(str(TUBE/"masks/outer"/SEQ/f"{f['rt']}.png"),cv2.IMREAD_GRAYSCALE)
        distance = cv2.distanceTransform((outer==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        blocks = {}
        row = {}
        for label in np.unique(ids[region]):
            sel = region & (ids==label)
            blocks[int(label)] = counts(q["cell"][sel])
            occ = sel & (q["cell"]//N<2)
            row[str(label)] = dict(occupancy=int(occ.sum()), predicted_body=int(q["predicted_body"][occ].sum()),
                                   measured_body=int(q["measured_body"][occ].sum()))
            if occ.any():
                if i < len(frames)-1:
                    edge_distances.setdefault(str(label),[]).extend(distance[q["py"][occ],q["px"][occ]].tolist())
                for k in ["pred","ref","ref_forward_m","ref_lateral_m","ref_height_m"]:
                    row[str(label)][k+"_p10_p50_p90"] = np.percentile(q[k][occ],[10,50,90]).tolist()
        frame_blocks.append(blocks)
        detail.append(dict(rgb_ts=f["rt"],dt_s=(f["ts"]-frames[-1]["ts"])/1e7,classes=row))
    roi_total = sum((sum(b.values()) for b in frame_blocks if b),np.zeros_like(total))
    all_ids=sorted(set().union(*(list(b) for b in frame_blocks)))
    result = {}
    for key, keep_history in [("all",False),("history_only",True)]:
        used=frame_blocks[:-1] if keep_history else frame_blocks
        grouped = {k:sum((b[k] for b in used if k in b),np.zeros_like(total)) for k in all_ids}
        def response(keys):
            group=sum((grouped[k] for k in keys),np.zeros_like(total))
            return dict(**paired_scores(group), roi_keep_without=paired_scores(roi_total-group)["keep"])
        result[key] = {str(k):response([k]) for k in all_ids}
        # Only seven classes send any evidence into this crop. Keep the upper uncertain
        # surface separate from both the traced lip and the lower, clearly visible wall.
        choices=[str(k) for k in all_ids]
        parts={str(k):[k] for k in all_ids}
        unions={}
        for bits in range(1,2**len(choices)):
            groups=[k for j,k in enumerate(choices) if bits & (1<<j)]
            selected=[kk for k in groups for kk in parts[k]]
            unions["+".join(groups)]=response(selected)
        result[key+"_combinations"]=unions
    edge_summary = {key:dict(n=len(values), within10_px=int((np.array(values)<=10).sum()),
                             p10_p50_p90_px=np.percentile(values,[10,50,90]).tolist())
                    for key,values in edge_distances.items()}
    return dict(class_names=names,frames=detail,checks=result,history_occ_distance_to_original_outline=edge_summary)


def focus_panel(case, frames, raw):
    """Show the preceding native exposure, explicitly marking only retained occupancy endpoints."""
    from PIL import Image, ImageDraw, ImageFont
    fontfile = next((p for p in [Path("C:/Windows/Fonts/msyh.ttc"),Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")] if p.exists()),None)
    if fontfile is None:
        return
    font=ImageFont.truetype(str(fontfile),18)
    small=ImageFont.truetype(str(fontfile),15)
    f,q=frames[-2],raw[-2]
    dt=(f["rt"]-frames[-1]["rt"])/1e7
    im=cv2.imread(str(DATA/"rgb"/f"{f['rt']}.png"))
    area=(q["px"]>=380) & (q["px"]<760) & (q["py"]>=0) & (q["py"]<107)
    keep=area & ~q["tube"] & (q["cell"]//N<2)
    left=im.copy()
    cv2.rectangle(left,(380,0),(759,106),(100,255,255),3)
    marked=im.copy()
    real_mask = cv2.imread(str(TUBE/"masks/outer"/SEQ/f"{f['rt']}.png"),cv2.IMREAD_GRAYSCALE)
    contours,_=cv2.findContours(real_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked,contours,-1,(255,237,67),1)
    label_path = ROOT.parent/"scene/labels"/f"{f['rt']}.png"
    label_im = cv2.imread(str(label_path),cv2.IMREAD_GRAYSCALE)
    if label_im is None:
        return
    colors_rgb = {14:(255,93,44),15:(178,88,255),17:(39,210,109)}
    for x,y in zip(q["px"][keep],q["py"][keep]):
        label = int(label_im[y,x])
        cv2.circle(marked,(x,y),2,colors_rgb.get(label,(95,95,95))[::-1],-1)
    panel=Image.new("RGB",(1024,400),(26,32,41))
    draw=ImageDraw.Draw(panel)
    draw.text((12,6),f"回程第{case-6}个报警帧   ·   取其前一张曝光（{dt:.2f} 秒）",font=font)
    thumb=Image.fromarray(cv2.cvtColor(left,cv2.COLOR_BGR2RGB)).resize((468,264),Image.Resampling.LANCZOS)
    panel.paste(thumb,(12,56))
    for j,crop in enumerate([im[:107,380:],marked[:107,380:]]):
        panel.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((520,146),Image.Resampling.NEAREST),
                    (492,39+151*j))
    draw.text((12,324),"黄框为右侧放大区，上图保留原始像素。",font=small,fill=(229,235,240))
    draw.text((492,338),f"青线为已撤去的管轮廓。下图标示仍被计入的 {int(keep.sum())} 个占据端点。",font=small,fill=(229,235,240))
    for x, key, word in [(12,14,"远处走廊投影区"),(224,15,"浅色斜边"),(393,17,"斜边上方，材质未定"),(681,0,"其他投影区")]:
        draw.ellipse((x,372,x+10,382),fill=colors_rgb.get(key,(95,95,95)))
        draw.text((x+17,366),word,font=small,fill=(229,235,240))
    panel.save(ROOT/f"focus-case{case}.jpg",quality=94)


def analyze(case, frames, floor, saved, net):
    x = original.features(frames,floor,V)
    raw = [events(f,frames[-1],floor,j==len(frames)-1) for j,f in enumerate(frames)]
    base = sum(counts(q["cell"]) for q in raw)
    if not np.array_equal(features(base,x),x):
        raise ValueError("Ray accounting did not reproduce the original input channels")
    off = [counts(q["cell"][~q["tube"]]) for q in raw]
    total = sum(off)
    baseline = score(net,features(total,x))
    if abs(baseline-saved["arms"]["outer"]["scores"]["all"])>1e-4:
        raise ValueError("Post-tube replay differs")
    def both(group):
        return dict(drop=score(net,features(total-group,x)), keep=score(net,features(group,x)))
    output = dict(post_tube_score=baseline, raw_score=score(net,x),
                  empty=score(net,features(total*0,x)), frames=[f["rt"] for f in frames], scans={})
    for ny,nx in [(4,4),(8,8)]:
        data = np.zeros((len(raw),ny*nx,*total.shape),dtype=np.int32)
        for j,q in enumerate(raw):
            use = ~q["tube"]
            tile = np.clip(q["py"],0,427)*ny//428*nx + np.clip(q["px"],0,759)*nx//760
            for tid in range(ny*nx):
                data[j,tid] = counts(q["cell"][use & (tile==tid)])
        if not np.array_equal(data.sum((0,1)),total):
            raise ValueError("Tile partition did not conserve evidence")
        tile_scores = {}
        for tid,block in enumerate(data.sum(0)):
            r = both(block)
            if (ny==4):
                r["by_source"] = [both(c) for c in data[:,tid]]
            tile_scores[tid] = r
        output["scans"][str(ny)] = tile_scores
    output["upper_region_checks"] = upper_region_checks(frames,raw,both)
    if (ROOT.parent/"scene/manifest.json").is_file():
        output["right_upper_by_scene"] = focused_scene(frames,raw,total,both)
    focus_panel(case,frames,raw)
    draw(frames,raw,case)
    for f,q in zip(frames,raw):
        np.savez_compressed(ROOT/f"events-{case}-{f['rt']}.npz", **q)
    return output


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    net = I.D.make_head(8)
    weights = WORK/"ase-cane-reach-pilot-20261015/decision_head8_all.pt"
    net.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    net.eval()
    saved = json.loads((TUBE/"d8/attribution.json").read_bytes())
    if original.file_digest(weights)!=saved["model_sha256"] or original.file_digest(TUBE/"masks/manifest.json")!=saved["mask_manifest_sha256"]:
        raise ValueError("Frozen evidence identity mismatch")
    ref = {r["case"]:r for r in saved["results"]}
    cap=original.Recording(DATA)
    geom={r["depth_ts"]:r for r in original.records(DATA/"geometry.jsonl")}
    frames=[]
    for r in json.loads((TUBE/"masks/histories.json").read_bytes())[SEQ]:
        dt,rt=r["depth_ts"],r["rgb_ts"]
        f=replay.cached_frame(cap,r,CACHE/"predicted"/f"{dt}.npy",A.MAX_MM*.001)
        mask=cv2.imread(str(TUBE/"masks/outer"/SEQ/f"{rt}.png"),0)
        f["tube"]=cv2.remap(mask,*f["maps"],cv2.INTER_NEAREST)>0
        f["rt"]=rt
        frames.append(f)
    result={}
    for case in (7,8):
        dt=ref[case]["depth_ts"]
        window=[f for f in frames if dt-original.WINDOW<=f["ts"]<=dt]
        result[case]=analyze(case,window,geom[dt]["floor_y"],ref[case],net)
    if all((ROOT/f"focus-case{case}.jpg").is_file() for case in (7,8)):
        panels=[cv2.imread(str(ROOT/f"focus-case{case}.jpg")) for case in (7,8)]
        cv2.imwrite(str(ROOT/"two-prior-exposures.jpg"),np.vstack(panels))
    (ROOT/"result.json").write_text(json.dumps(result,indent=1,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
