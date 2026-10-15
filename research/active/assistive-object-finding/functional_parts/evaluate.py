#!/usr/bin/env python3
"""CPU-only functional-part Development evaluation and calibration-only head choice.

Head selection accepts a dedicated AGD calibration manifest, not a complete
evaluation manifest. AGD continuous heatmaps and IIT categorical regions retain
their native interpretation. No ground truth enters map or point selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label as connected_components
from scipy.spatial import cKDTree
cv2.setNumThreads(2)

HEAD_ORDER = ("raw", "ree", "rem")
METHODS = ("M1_base_argmax", "M2_base_weighted_centroid", "M3_base_component_DT",
           "M4_base_flip_mean", "M5_base_center_crop_mean", "M6_base_peak_crop_mean")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8 << 20),b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path,rows):
    Path(path).write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n" for row in rows),encoding="utf-8")


def is_agd(row):
    return "agd" in str(row["dataset"]).lower()


def split_name(row):
    value=str(row["split"]).lower()
    return "calibration" if value in ("calibration","cal") else "evaluation" if value in ("evaluation","eval") else value


def load_map(path):
    array=np.asarray(np.load(path,allow_pickle=False)).squeeze()
    if array.ndim!=2 or not np.isfinite(array).all():
        raise ValueError(f"Expected finite 2D raw heatmap: {path}")
    return array.astype(np.float32,copy=False)


def official_minmax(raw,size_wh):
    """Match released normalize_map: cv2 linear resize, minmax, eps=1e-10."""
    value=cv2.resize(np.asarray(raw,dtype=np.float32),tuple(map(int,size_wh)),interpolation=cv2.INTER_LINEAR)
    lo,hi=value.min(),value.max()
    return (value-lo)/(hi-lo+1e-10)


def probability(raw,size_wh):
    heatmap=official_minmax(raw,size_wh).astype(np.float64)
    total=float(heatmap.sum())
    return heatmap/total if total>0 else np.zeros_like(heatmap), total<=0


def native_heatmap_metrics(prediction,gt):
    """Released author KLD/SIM/NSS formulas; undefined NSS remains None."""
    prediction=np.asarray(prediction)
    gt=np.asarray(gt)
    eps=1e-12
    p=prediction/(prediction.sum()+eps)
    q=gt/(gt.sum()+eps)
    kld=float(np.sum(q*np.log(q/(p+eps)+eps)))
    sim=float(np.minimum(p,q).sum())
    std=float(np.std(prediction/255.))
    fixation=((gt-gt.min())/(gt.max()-gt.min()+eps))>.1
    if std<=0 or not fixation.any():
        nss=None
    else:
        z=(prediction/255.-np.mean(prediction/255.))/std
        nss=float(np.sum(z*fixation)/np.sum(fixation+eps))
    return {"KLD":kld,"SIM":sim,"NSS":nss,"NSS_evaluable":nss is not None,
            "zero_gt_mass":bool(gt.sum()<=0),"constant_prediction":bool(prediction.max()==prediction.min())}


def native_agd_from_raw(raw,gt_raw):
    pred=official_minmax(raw,(224,224))
    gt=cv2.resize(np.asarray(gt_raw,dtype=np.float64)/255.,(224,224),interpolation=cv2.INTER_LINEAR)
    return native_heatmap_metrics(pred,gt)


def normalize_mass(array):
    value=np.asarray(array,dtype=np.float64)
    total=float(value.sum())
    return value/total if total>0 else np.zeros_like(value)


def argmax_point(array):
    y,x=np.unravel_index(int(np.argmax(array)),array.shape)
    return [float(x)+.5,float(y)+.5]


def point_pixel(point,shape):
    # Point coordinates refer to pixel centers. Exact midpoint ties go to the
    # smaller index, matching the row-major tie convention.
    x,y=point
    return max(0,min(shape[1]-1,int(math.ceil(x-1.)))),max(0,min(shape[0]-1,int(math.ceil(y-1.))))


def weighted_centroid(prob):
    h,w=prob.shape
    if prob.sum()<=0:
        return [w/2.,h/2.],"ZERO_MASS_CENTER_FALLBACK"
    return [float((prob.sum(axis=0)*(np.arange(w)+.5)).sum()),
            float((prob.sum(axis=1)*(np.arange(h)+.5)).sum())],None


def component_interior(prob):
    peak=argmax_point(prob)
    if prob.max()<=0:
        return peak,"EMPTY_COMPONENT_ARGMAX_FALLBACK"
    support=prob>=.5*prob.max()
    components,_=connected_components(support)  # fixed scipy 4-neighbour default
    x,y=point_pixel(peak,prob.shape)
    component=components==components[y,x]
    if components[y,x]==0 or not component.any():
        return peak,"EMPTY_COMPONENT_ARGMAX_FALLBACK"
    # Image edges are treated as boundaries as well as the complement.
    distances=distance_transform_edt(np.pad(component,1,constant_values=False))[1:-1,1:-1]
    return argmax_point(distances),None


def projected_crop(raw,bbox,size_wh):
    w,h=map(int,size_wh)
    if len(bbox)!=4 or any(int(v)!=v for v in bbox):
        raise ValueError("Crop bbox must contain integer half-open pixel coordinates")
    x0,y0,x1,y1=map(int,bbox)
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):
        raise ValueError("Crop bbox outside original image")
    local,degenerate=probability(raw,(x1-x0,y1-y0))
    result=np.zeros((h,w),dtype=np.float64)
    result[y0:y1,x0:x1]=local
    return result,degenerate


def spatial_confidence(prob):
    if prob.sum()<=0: return None
    small=cv2.resize(prob,(64,64),interpolation=cv2.INTER_LINEAR)
    small=normalize_mass(small)
    if small.sum()<=0:return None
    positive=small[small>0]
    entropy=float(-np.sum(positive*np.log(positive)))
    return float(np.clip(1.-entropy/math.log(4096.),0.,1.))


def separated_peak_margin(prob):
    if prob.max()<=0:return None
    h,w=prob.shape
    peak=argmax_point(prob)
    yy,xx=np.ogrid[:h,:w]
    outside=(xx+.5-peak[0])**2+(yy+.5-peak[1])**2>(.1*math.hypot(w,h))**2
    second=float(prob[outside].max()) if outside.any() else 0.
    return float((prob.max()-second)/(prob.max()+1e-12))


def bilinear_value(array,point):
    h,w=array.shape
    x=float(np.clip(point[0]-.5,0,w-1));y=float(np.clip(point[1]-.5,0,h-1))
    x0,y0=int(math.floor(x)),int(math.floor(y))
    x1,y1=min(x0+1,w-1),min(y0+1,h-1)
    dx,dy=x-x0,y-y0
    return float(array[y0,x0]*(1-dx)*(1-dy)+array[y0,x1]*dx*(1-dy)+array[y1,x0]*(1-dx)*dy+array[y1,x1]*dx*dy)


def gt_geometry(weights):
    mass=float(weights.sum())
    if mass<=0:return {"effective_area_fraction":None,"elongation":None}
    h,w=weights.shape
    px=weights.sum(axis=0)/mass;py=weights.sum(axis=1)/mass
    xs=np.arange(w)+.5;ys=np.arange(h)+.5
    cx=float(px@xs);cy=float(py@ys)
    vx=float(px@((xs-cx)**2));vy=float(py@((ys-cy)**2))
    cov=float(((weights@(xs-cx))*(ys-cy)).sum()/mass)
    eig=np.linalg.eigvalsh([[vx,cov],[cov,vy]])
    elongation=float(math.sqrt(max(0.,eig[1])/max(1e-12,eig[0])))
    return {"effective_area_fraction":float(mass*mass/((weights*weights).sum()*h*w)),"elongation":elongation}


def agd_point_metrics(gt,point,peak_tree):
    mass=float(gt.sum());maximum=float(gt.max());h,w=gt.shape
    if maximum<=0 or mass<=0:
        return {"GT_relative_peak_proxy":None,"local_GT_soft_mass_proxy":None,"GT_peak_distance_diag":None}
    radius=.05*math.hypot(w,h)
    x0=max(0,int(math.floor(point[0]-radius-.5)));x1=min(w,int(math.ceil(point[0]+radius-.5))+1)
    y0=max(0,int(math.floor(point[1]-radius-.5)));y1=min(h,int(math.ceil(point[1]+radius-.5))+1)
    yy,xx=np.ogrid[y0:y1,x0:x1]
    disk=(xx+.5-point[0])**2+(yy+.5-point[1])**2<=radius*radius
    return {"GT_relative_peak_proxy":bilinear_value(gt,point)/maximum,
        "local_GT_soft_mass_proxy":float((gt[y0:y1,x0:x1]*disk).sum()/mass),
        "GT_peak_distance_diag":float(peak_tree.query(point)[0]/math.hypot(w,h))}


def iit_point_metrics(gt,point,boundary):
    x,y=point_pixel(point,gt.shape)
    label=int(gt[y,x])
    return {"native_union_hit":label in (5,9),"native_grasp5_hit":label==5,
        "native_wrap9_hit":label==9,"native_point_label":label,
        "signed_union_boundary_distance_diag":float(boundary[y,x]/math.hypot(*gt.shape)) if boundary is not None else None}


def make_methods(base_raw,additional,size_wh):
    """Observations only. No GT or dataset object annotations accepted."""
    w,h=map(int,size_wh)
    base,base_degenerate=probability(base_raw,size_wh)
    flip_raw=load_map(additional["flip_map_path"])
    if not additional.get("flip_map_is_inverse_restored",False):
        raise ValueError("Require explicitly inverse-restored horizontal-flip map")
    flip,flip_degenerate=probability(flip_raw,size_wh)
    center,center_degenerate=projected_crop(load_map(additional["fixed_crop_map_path"]),additional["fixed_crop_bbox_xyxy"],size_wh)
    peak,peak_degenerate=projected_crop(load_map(additional["peak_crop_map_path"]),additional["peak_crop_bbox_xyxy"],size_wh)
    if (additional["fixed_crop_bbox_xyxy"][2]-additional["fixed_crop_bbox_xyxy"][0],additional["fixed_crop_bbox_xyxy"][3]-additional["fixed_crop_bbox_xyxy"][1]) != \
       (additional["peak_crop_bbox_xyxy"][2]-additional["peak_crop_bbox_xyxy"][0],additional["peak_crop_bbox_xyxy"][3]-additional["peak_crop_bbox_xyxy"][1]):
        raise ValueError("Fixed and guided crops must have equal sizes")
    maps={name:base for name in METHODS[:3]}
    maps.update({METHODS[3]:normalize_mass(.5*base+.5*flip),METHODS[4]:normalize_mass(.5*base+.5*center),METHODS[5]:normalize_mass(.5*base+.5*peak)})
    centroid,centroid_fallback=weighted_centroid(base)
    interior,interior_fallback=component_interior(base)
    points={name:argmax_point(value) for name,value in maps.items()}
    points[METHODS[1]]=centroid;points[METHODS[2]]=interior
    confidence={"G1_M1_entropy":spatial_confidence(base),
        "G2_M1_flip_consistency":None if base_degenerate or flip_degenerate else
            float(1.-math.dist(points[METHODS[0]],argmax_point(flip))/math.hypot(w,h)),
        "G3_M6_entropy":spatial_confidence(maps[METHODS[5]])}
    flags={"base_component_degenerate":base_degenerate,"flip_component_degenerate":flip_degenerate,
        "fixed_crop_component_degenerate":center_degenerate,"peak_crop_component_degenerate":peak_degenerate,
        "centroid_fallback":centroid_fallback,"DT_fallback":interior_fallback,
        "runner_peak_crop_fallback":additional.get("peak_crop_fallback")}
    return maps,points,confidence,flags,separated_peak_margin(base)


def finite_mean(values,total):
    valid=[float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {"mean":float(np.mean(valid)) if valid else None,"n_evaluable":len(valid),
            "image_denominator":int(total),"NOT_EVALUABLE":int(total-len(valid))}


def summarize(records,dataset):
    n=len(records)
    result={"images":n,"unique_original_image_sha256":len({r["image_sha256"] for r in records})}
    if dataset=="AGD20K":
        for key in ("KLD","SIM","NSS","GT_relative_peak_proxy","local_GT_soft_mass_proxy","GT_peak_distance_diag"):
            result[key]=finite_mean([r.get(key) for r in records],n)
        result["mean_one_minus_relative_peak_risk_proxy"]=finite_mean([None if r.get("GT_relative_peak_proxy") is None else 1.-r["GT_relative_peak_proxy"] for r in records],n)
    else:
        for key in ("native_union_hit","native_grasp5_hit","native_wrap9_hit"):
            result[key]={"n":sum(bool(r[key]) for r in records),"d":n,
                "value":sum(bool(r[key]) for r in records)/n if n else None}
        result["native_nonempty_images"]=sum(not r["native_empty"] for r in records)
        result["native_empty_images"]=sum(r["native_empty"] for r in records)
        nonempty=[r for r in records if not r["native_empty"]]
        result["native_nonempty_union_hit"]={"n":sum(r["native_union_hit"] for r in nonempty),"d":len(nonempty),
            "value":sum(r["native_union_hit"] for r in nonempty)/len(nonempty) if nonempty else None}
        for label,key in ((5,"native_grasp5_hit"),(9,"native_wrap9_hit")):
            present=[r for r in records if r["native_label_pixel_counts"][str(label)]>0]
            result[key+"_label_present_only"]={"n":sum(r[key] for r in present),"d":len(present),
                "value":sum(r[key] for r in present)/len(present) if present else None}
        result["native_union_miss_risk"]={"n":sum(not r["native_union_hit"] for r in records),"d":n,
            "value":sum(not r["native_union_hit"] for r in records)/n if n else None}
        result["signed_union_boundary_distance_diag"]=finite_mean([r.get("signed_union_boundary_distance_diag") for r in records],n)
    result["base_degenerate_images"]=sum(r["flags"]["base_component_degenerate"] for r in records)
    return result


def calibrate_coverage(scores,total):
    required=int(math.ceil(.8*total))
    valid=sorted((float(s) for s in scores if s is not None and math.isfinite(float(s))),reverse=True)
    reachable=len(valid)>=required
    threshold=valid[required-1] if reachable else min(valid) if valid else None
    accepted=sum(s>=threshold for s in valid) if threshold is not None else 0
    return {"threshold":threshold,"required_count":required,"calibration_images":total,
        "calibration_finite_scores":len(valid),"calibration_accepted":accepted,
        "target_reachable":reachable,"status":"SEALED" if reachable else "TARGET_UNREACHABLE",
        "rule":"highest threshold retaining>=ceil(.8*N); ties accepted; nonfinite rejected"}


def paired_summary(reference,trial,dataset):
    old={r["image_sha256"]:r for r in reference};new={r["image_sha256"]:r for r in trial}
    if set(old)!=set(new):raise ValueError("Paired source sets mismatch")
    result={"paired_original_images":len(old)}
    keys=("KLD","SIM","NSS","GT_relative_peak_proxy","local_GT_soft_mass_proxy","GT_peak_distance_diag") if dataset=="AGD20K" else ("native_union_hit","native_grasp5_hit","native_wrap9_hit")
    for key in keys:
        deltas=[float(new[k][key])-float(old[k][key]) for k in sorted(old)
                if old[k].get(key) is not None and new[k].get(key) is not None]
        lower_better=key in ("KLD","GT_peak_distance_diag")
        result[key]={"delta_trial_minus_reference_mean":float(np.mean(deltas)) if deltas else None,
            "n_evaluable":len(deltas),"image_denominator":len(old),
            "improved":sum(d < -1e-12 if lower_better else d > 1e-12 for d in deltas),
            "worsened":sum(d > 1e-12 if lower_better else d < -1e-12 for d in deltas),
            "tied":sum(abs(d)<=1e-12 for d in deltas)}
        if dataset!="AGD20K":
            result[key]["saved"]=result[key]["improved"];result[key]["harmed"]=result[key]["worsened"]
    return result


def evaluate(args):
    started=time.monotonic();cpu_started=time.process_time()
    root=Path(args.root).resolve()
    manifest=args.manifest or root/"data/samples.jsonl"
    base_path=args.predictions or root/"inference/base_predictions.jsonl"
    additional_path=args.additional_predictions or root/"additional/predictions.jsonl"
    head_path=args.sealed_head or root/"analysis_head/sealed_head.json"
    out=args.output_dir or root/"analysis"
    out.mkdir(parents=True,exist_ok=True)
    if (out/"summary.json").exists() or (out/"sealed_gates.json").exists():
        raise FileExistsError("Analysis sealed/completed; no evaluation retuning or overwrite")
    plan=Path(args.plan)
    head=json.loads(Path(head_path).read_text(encoding="utf-8"))
    if head["base_predictions_sha256"]!=sha256(base_path) or head["plan_sha256"]!=sha256(plan):
        raise ValueError("Sealed head input/PLAN changed")
    primary=head["selected_head"]
    if primary not in HEAD_ORDER or head["evaluation_used"]:
        raise ValueError("Invalid calibration-only selected head")
    snapshot=out/"evaluate_at_scoring.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    rows=read_jsonl(manifest)
    if len(rows)!=280 or len({r["sample_id"] for r in rows})!=280 or len({r["image_sha256"] for r in rows})!=280:
        raise ValueError("Expected280 unique primary samples/images; repeated queries excluded")
    for row in rows:
        row["dataset_key"]="AGD20K" if is_agd(row) else "IIT-AFF" if "iit" in str(row["dataset"]).lower() else "UNKNOWN"
        row["phase"]=split_name(row)
    counts=Counter((r["dataset_key"],r["phase"]) for r in rows)
    expected=Counter({("AGD20K","calibration"):80,("AGD20K","evaluation"):160,("IIT-AFF","calibration"):16,("IIT-AFF","evaluation"):24})
    if counts!=expected:raise ValueError(f"Unexpected dataset/split counts {counts}")
    cal=[r for r in rows if r["phase"]=="calibration"]
    evaluation=[r for r in rows if r["phase"]=="evaluation"]
    if {r["image_sha256"] for r in cal}&{r["image_sha256"] for r in evaluation}:
        raise ValueError("Original-image SHA crosses calibration/evaluation")
    cal_selector=read_jsonl(head["calibration_manifest"])
    if {r["sample_id"] for r in cal_selector}!={r["sample_id"] for r in cal if r["dataset_key"]=="AGD20K"}:
        raise ValueError("Head calibration set differs from current main calibration")
    inputs={"PLAN_sha256":sha256(plan),"manifest_sha256":sha256(manifest),"base_predictions_sha256":sha256(base_path),
        "additional_predictions_sha256":sha256(additional_path),"sealed_head_sha256":sha256(head_path),
        "scoring_script_sha256":sha256(__file__),"snapshot_sha256":sha256(snapshot),
        "released_native_metric_source_sha256":sha256(root/"vendor/SelectiveCL/utils/evaluation.py")}
    base_list=read_jsonl(base_path);additional_list=read_jsonl(additional_path)
    base={r["sample_id"]:r for r in base_list};additional={r["sample_id"]:r for r in additional_list}
    ids={r["sample_id"] for r in rows}
    if len(base_list)!=280 or len(additional_list)!=280 or set(base)!=ids or set(additional)!=ids:
        raise ValueError("Inference records incomplete/duplicate/extra")
    split_hashes={}
    for dataset in ("AGD20K","IIT-AFF"):
        for phase in ("calibration","evaluation"):
            subset=sorted([r for r in rows if r["dataset_key"]==dataset and r["phase"]==phase],key=lambda r:r["sample_id"])
            path=out/(dataset.replace('-','_')+"_"+phase+"_manifest.jsonl")
            write_jsonl(path,subset);split_hashes[dataset+"_"+phase]=sha256(path)
    records=[];head_diagnostics=[];source_details={};map_hashes=[]

    def process_one(row):
        if time.process_time()-cpu_started>600:raise TimeoutError("Evaluation CPU600s budget exhausted")
        sid=row["sample_id"];observation=base[sid];views=additional[sid]
        size=(int(row["width"]),int(row["height"]))
        for obs in (observation,views):
            if obs["image_sha256"]!=row["image_sha256"] or obs["action"]!=row["action"] or (int(obs["width"]),int(obs["height"]))!=size:
                raise ValueError("Inference/evaluation image or query identity mismatch")
        if views["primary_head"]!=primary:
            raise ValueError("Additional views use another head")
        base_raw=load_map(get_base_head_path(observation,primary))
        if views["base_map_path"]!=get_base_head_path(observation,primary):
            raise ValueError("Additional stage base map differs")
        maps,points,confidence,flags,margin=make_methods(base_raw,views,size)
        # All inference/map/point/gate operations precede this sole GT access.
        gt=cv2.imread(str(row["gt_path"]),cv2.IMREAD_GRAYSCALE)
        if gt is None or gt.shape!=(size[1],size[0]):
            raise ValueError("GT decode/shape mismatch")
        if row.get("gt_sha256") and sha256(row["gt_path"])!=row["gt_sha256"]:
            raise ValueError("GT content changed")
        tree=None;boundary=None
        if row["dataset_key"]=="AGD20K":
            gt=gt.astype(np.float64)
            yy,xx=np.where(gt==gt.max())
            tree=cKDTree(np.column_stack([xx+.5,yy+.5])) if gt.max()>0 else None
            geom=gt_geometry(gt)
            native=native_agd_from_raw(base_raw,gt)
            for branch in HEAD_ORDER:
                value=native if branch==primary else native_agd_from_raw(load_map(get_base_head_path(observation,branch)),gt)
                head_diagnostics.append({"sample_id":sid,"image_sha256":row["image_sha256"],"phase":row["phase"],"head":branch,**value})
            native_empty=None
        else:
            if any(int(v) not in range(10) for v in np.unique(gt)):
                raise ValueError("IIT labels outside0..9")
            if row.get("target_label_ids")!=[5,9]:raise ValueError("IIT target union must be native5/9")
            region=np.isin(gt,[5,9]);native_empty=not bool(region.any())
            if row.get("native_empty") is not None and row["native_empty"]!=native_empty:
                raise ValueError("IIT native-empty metadata mismatch")
            geom=gt_geometry(region.astype(float))
            if not native_empty:
                inside=distance_transform_edt(np.pad(region,1,constant_values=False))[1:-1,1:-1]
                outside=distance_transform_edt(~region)
                boundary=np.where(region,inside,-outside)
            native=None
        detail={"sample_id":sid,"image_sha256":row["image_sha256"],"dataset":row["dataset_key"],"phase":row["phase"],
            "action":row["action"],"object":row["object"],"geometry":geom,"confidence":confidence,
            "base_peak_margin_diagnostic":margin,"flags":flags,"native_empty":native_empty,
            "fixed_crop_bbox_xyxy":views["fixed_crop_bbox_xyxy"],"peak_crop_bbox_xyxy":views["peak_crop_bbox_xyxy"],
            "selected_head":primary,"GT_kind":"continuous_heatmap" if row["dataset_key"]=="AGD20K" else "native_categorical_union5_9"}
        if row["dataset_key"]=="IIT-AFF":
            detail["native_label_pixel_counts"]={str(label):int((gt==label).sum()) for label in (5,9)}
            detail["native_union_pixels"]=sum(detail["native_label_pixel_counts"].values())
        source_details[sid]=detail
        elapsed=views["elapsed"]
        for j,name in enumerate(METHODS):
            point=points[name]
            metric=agd_point_metrics(gt,point,tree) if row["dataset_key"]=="AGD20K" else iit_point_metrics(gt,point,boundary)
            if row["dataset_key"]=="AGD20K":
                hm=native if j<3 else native_agd_from_raw(maps[name],gt)
                metric={**hm,**metric}
            view_name="base" if j<3 else ("flip","fixed_crop","peak_crop")[j-3]
            inference_s=float(elapsed["base"]["actual_view_s"])
            if j>=3:inference_s+=float(elapsed[view_name]["actual_view_s"])
            records.append({**detail,"method":name,"point_xy_pixel_centers":point,
                "point_pixel_nearest_center_lower_tie":list(point_pixel(point,gt.shape)),
                "model_calls_per_configuration":1 if j<3 else 2,"measured_view_inference_s":inference_s,
                "native_resampling_path":"raw14_to224_author" if j<3 else "original_image_probability_fusion_to224_author",
                **metric})
        for map_name,path in (("base",get_base_head_path(observation,primary)),("flip",views["flip_map_path"]),
                              ("fixed_crop",views["fixed_crop_map_path"]),("peak_crop",views["peak_crop_map_path"])):
            map_hashes.append({"sample_id":sid,"view":map_name,"sha256":sha256(path)})

    # Calibration is scored first; thresholds/bins are sealed before eval GT opens.
    for row in sorted(cal,key=lambda r:r["sample_id"]):process_one(row)
    gates={};strata_cutoffs={}
    gate_methods={"G1_M1_entropy":METHODS[0],"G2_M1_flip_consistency":METHODS[0],"G3_M6_entropy":METHODS[5]}
    for dataset in ("AGD20K","IIT-AFF"):
        details=[d for d in source_details.values() if d["dataset"]==dataset]
        gates[dataset]={gate:calibrate_coverage([d["confidence"][gate] for d in details],len(details)) for gate in gate_methods}
        strata_cutoffs[dataset]={key:float(np.median([d["geometry"][key] for d in details if d["geometry"][key] is not None]))
            if any(d["geometry"][key] is not None for d in details) else None for key in ("effective_area_fraction","elongation")}
    sealed={"status":"SEALED","selected_released_head":primary,"gates":gates,"gate_methods":gate_methods,
        "calibration_GT_strata_medians_evaluator_only":strata_cutoffs,"inputs":inputs,"split_manifest_sha256":split_hashes,
        "coverage_rule":"dataset-specific80percent calibration; ties accepted; no eval retention guarantee",
        "point_radius_diag":.05,"DT_relative_peak_threshold":.5,"DT_connectivity":4,
        "no_GT_in_point_or_confidence":True,"evaluation_scored_before_sealing":False}
    write_json(out/"sealed_gates.json",sealed)
    gate_sha=sha256(out/"sealed_gates.json")
    for row in sorted(evaluation,key=lambda r:r["sample_id"]):process_one(row)
    write_jsonl(out/"per_image_method_metrics.jsonl",records)
    write_jsonl(out/"per_image_confidence_geometry.jsonl",[source_details[k] for k in sorted(source_details)])
    write_jsonl(out/"all_head_baseline_native_diagnostic.jsonl",head_diagnostics)
    summary={"status":"COMPLETE","evidence":"Development; AGD continuous-heatmap proxies, IIT native categorical region diagnostic",
        "selected_head":primary,"head_selection":"80AGDcal native meanKLD; not evaluation-selected or exact paper-table reproduction",
        "original_images":280,"datasets":{},"paired_evaluation":{},"gates_evaluation":{},"error_strata_evaluation":{},
        "inputs":inputs,"sealed_gates_sha256":gate_sha,"split_manifest_sha256":split_hashes,
        "interpretation_limits":["AGDpoint values/softmass are continuous heatmap proxies, not mask hits",
            "IITgrasp/wrapgrasp hits are image-plane semantic matches, not safe contact or task success",
            "Separate datasets; no combined score; calibration coverage not eval guarantee",
            "M6entropy system comparison is not isolated evidence for a new gate",
            "Only280primary queries; repeated-action queries share original-image denominator and were not added"]}
    gate_decisions=[]
    pairings=[(METHODS[0],name) for name in METHODS[1:]]+[(METHODS[4],METHODS[5]),(METHODS[3],METHODS[5])]
    for dataset in ("AGD20K","IIT-AFF"):
        summary["datasets"][dataset]={phase:{name:summarize([r for r in records if r["dataset"]==dataset and r["phase"]==phase and r["method"]==name],dataset)
            for name in METHODS} for phase in ("calibration","evaluation")}
        eval_records=[r for r in records if r["dataset"]==dataset and r["phase"]=="evaluation"]
        summary["paired_evaluation"][dataset]={trial+"_vs_"+ref:paired_summary([r for r in eval_records if r["method"]==ref],[r for r in eval_records if r["method"]==trial],dataset) for ref,trial in pairings}
        summary["gates_evaluation"][dataset]={}
        for gate,method in gate_methods.items():
            setting=gates[dataset][gate];threshold=setting["threshold"]
            subset=[r for r in eval_records if r["method"]==method]
            retained=[]
            for record in subset:
                confidence=record["confidence"][gate]
                accepted=confidence is not None and math.isfinite(confidence) and threshold is not None and confidence>=threshold
                gate_decisions.append({"sample_id":record["sample_id"],"image_sha256":record["image_sha256"],"dataset":dataset,
                    "gate":gate,"method":method,"phase":"evaluation","confidence":confidence,"threshold":threshold,
                    "accepted":accepted,"sealed_gates_sha256":gate_sha})
                if accepted:retained.append(record)
            summary["gates_evaluation"][dataset][gate]={"calibration":setting,
                "actual_evaluation_coverage":{"n":len(retained),"d":len(subset),"value":len(retained)/len(subset) if subset else None},
                "retained_quality":summarize(retained,dataset),"full_coverage_quality":summarize(subset,dataset)}
        strata={}
        for grouping in ("action","object","effective_area_fraction","elongation"):
            grouped=defaultdict(list)
            for r in eval_records:
                if grouping in ("action","object"):key=str(r[grouping])
                else:
                    value=r["geometry"][grouping];cut=strata_cutoffs[dataset][grouping]
                    key="NOT_EVALUABLE" if value is None or cut is None else "at_or_below_cal_median" if value<=cut else "above_cal_median"
                grouped[(key,r["method"])].append(r)
            strata[grouping]={}
            for (key,method),items in grouped.items():strata[grouping].setdefault(key,{})[method]=summarize(items,dataset)
        summary["error_strata_evaluation"][dataset]=strata
        for name in METHODS:
            times=[r["measured_view_inference_s"] for r in eval_records if r["method"]==name]
            summary["datasets"][dataset]["evaluation"][name]["model_view_timing_s"]={"mean":float(np.mean(times)),"median":float(np.median(times)),"p95":float(np.percentile(times,95)),"model_calls":1 if name in METHODS[:3] else 2,"scope":"measured preprocess+GPUforward for prescribed views; no end-to-end or user latency claim"}
    write_jsonl(out/"per_image_gate_decisions.jsonl",gate_decisions)
    head_summary={phase:{head_name:{key:finite_mean([r[key] for r in head_diagnostics if r["phase"]==phase and r["head"]==head_name],80 if phase=="calibration" else 160)
            for key in ("KLD","SIM","NSS")} for head_name in HEAD_ORDER} for phase in ("calibration","evaluation")}
    if abs(head_summary["calibration"][primary]["KLD"]["mean"]-head["mean_KLD"][primary])>1e-10:
        raise ValueError("Calibration native metric does not reproduce sealed head selection")
    summary["all_head_AGD_native_diagnostic_no_switching"]=head_summary
    summary["analysis_cpu_seconds"]=time.process_time()-cpu_started
    summary["analysis_wall_seconds"]=time.monotonic()-started
    summary["gpu_seconds"]=0;summary["new_download_bytes"]=0
    write_json(out/"summary.json",summary)
    write_json(out/"receipt.json",{"status":"COMPLETE","inputs":inputs,"snapshot":str(snapshot.resolve()),
        "outputs":{p.name:sha256(p) for p in out.iterdir() if p.is_file()},"raw_selected_map_hashes":map_hashes,
        "analysis_cpu_seconds":time.process_time()-cpu_started,"analysis_wall_seconds":time.monotonic()-started,
        "GPU_seconds":0,"download_bytes":0})
    print(json.dumps({"status":"COMPLETE","selected_head":primary,"analysis_cpu_seconds":summary["analysis_cpu_seconds"],
        "AGD_evaluation":summary["datasets"]["AGD20K"]["evaluation"],"IIT_evaluation":summary["datasets"]["IIT-AFF"]["evaluation"],
        "paired_M6_vs_M5_AGD":summary["paired_evaluation"]["AGD20K"][METHODS[5]+"_vs_"+METHODS[4]],
        "paired_M6_vs_M5_IIT":summary["paired_evaluation"]["IIT-AFF"][METHODS[5]+"_vs_"+METHODS[4]]},ensure_ascii=False))


def get_base_head_path(observation,head):
    if "base_heads" in observation:
        entry=observation["base_heads"][head]
    elif "heads" in observation and head in observation["heads"]:
        entry=observation["heads"][head]
    else:
        raise ValueError("Base predictions require base_heads raw/ree/rem map paths")
    return entry["map_path"] if isinstance(entry,dict) else entry


def select_head(args):
    """Only this dedicated calibration GT manifest is accepted by selector."""
    started=time.monotonic()
    manifest=Path(args.calibration_manifest)
    rows=read_jsonl(manifest)
    if len(rows)!=80 or any(not is_agd(row) or split_name(row)!="calibration" for row in rows):
        raise ValueError("Head selector requires exactly80 AGD calibration rows; evaluation/IIT GT forbidden")
    if len({row["sample_id"] for row in rows})!=80 or len({row["image_sha256"] for row in rows})!=80:
        raise ValueError("AGD calibration must have80 unique samples and original image hashes")
    plan=Path(args.plan)
    if not plan.is_file():
        raise FileNotFoundError("Frozen PLAN.md required before head selection")
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"sealed_head.json").exists():
        raise FileExistsError("Head already sealed; do not select again")
    snapshot=out/"evaluate_at_head_selection.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    # Prediction records are observations; filter to permitted calibration ids
    # before any GT is opened. Complete eval GT is never an argument here.
    allowed={row["sample_id"] for row in rows}
    observations={}
    for raw in read_jsonl(args.predictions):
        if raw["sample_id"] in allowed:
            if raw["sample_id"] in observations: raise ValueError("Duplicate calibration observation")
            observations[raw["sample_id"]]=raw
    if set(observations)!=allowed: raise ValueError("Calibration base inference incomplete")
    per_row=[]
    map_receipts=[]
    for row in sorted(rows,key=lambda r:r["sample_id"]):
        observation=observations[row["sample_id"]]
        if observation.get("image_sha256")!=row["image_sha256"]:
            raise ValueError("Calibration prediction image identity mismatch")
        gt=cv2.imread(str(row["gt_path"]),cv2.IMREAD_GRAYSCALE)
        if gt is None: raise ValueError(f"Cannot decode calibration GT {row['gt_path']}")
        for head in HEAD_ORDER:
            map_path=get_base_head_path(observation,head)
            result=native_agd_from_raw(load_map(map_path),gt)
            per_row.append({"sample_id":row["sample_id"],"image_sha256":row["image_sha256"],"head":head,**result})
            map_receipts.append({"sample_id":row["sample_id"],"head":head,"sha256":sha256(map_path)})
    mean_kld={head:float(np.mean([row["KLD"] for row in per_row if row["head"]==head])) for head in HEAD_ORDER}
    head=min(HEAD_ORDER,key=lambda name:(mean_kld[name],HEAD_ORDER.index(name)))
    result={"status":"SEALED","selected_head":head,"head_order":list(HEAD_ORDER),
        "criterion":"lowest mean native224 KLD on80 unique AGD calibration images; exact tie order raw,ree,rem",
        "calibration_count":80,"mean_KLD":mean_kld,"evaluation_used":False,"IIT_selection_used":False,
        "label":"calibration-selected released-code branch; not exact paper-table reproduction",
        "calibration_manifest":str(manifest.resolve()),"calibration_manifest_sha256":sha256(manifest),
        "base_predictions_sha256":sha256(args.predictions),"plan_sha256":sha256(plan),
        "selection_script_sha256":sha256(__file__),"selection_script_snapshot":str(snapshot.resolve()),
        "selection_script_snapshot_sha256":sha256(snapshot),"map_receipts":map_receipts,
        "wall_seconds":time.monotonic()-started,"gpu_seconds":0,"download_bytes":0}
    write_jsonl(out/"head_calibration_metrics.jsonl",per_row)
    write_json(out/"sealed_head.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="map_receipts"},ensure_ascii=False))


def self_test(vendor_metrics=None):
    # Released normalization must preserve negative-logit ordering, not ReLU.
    raw=np.asarray([[-2.,-1.],[0.,2.]],dtype=np.float32)
    norm=official_minmax(raw,(2,2))
    assert np.allclose(norm,[[0.,.25],[.5,1.]])
    p,degenerate=probability(raw,(2,2))
    assert not degenerate and abs(p.sum()-1)<1e-12
    result=native_heatmap_metrics(np.asarray([[1.,0.],[0.,1.]]),np.asarray([[1.,0.],[0.,1.]]))
    assert abs(result["KLD"])<1e-9 and abs(result["SIM"]-1)<1e-9 and abs(result["NSS"]-1)<1e-9
    p,degenerate=probability(np.ones((2,2)),(2,2))
    assert degenerate and p.sum()==0
    assert native_heatmap_metrics(p,np.eye(2))["NSS"] is None
    crop,crop_flat=projected_crop(np.asarray([[0.,1.],[0.,1.]]),[2,1,4,3],(6,5))
    assert not crop_flat and abs(crop.sum()-1)<1e-12
    assert crop[:1].sum()==0 and crop[3:].sum()==0 and crop[:,:2].sum()==0 and crop[:,4:].sum()==0
    block=np.zeros((7,7));block[2:5,2:5]=1./9
    assert component_interior(block)==([3.5,3.5],None)
    assert point_pixel([1.,1.],(3,3))==(0,0)
    assert calibrate_coverage([.9,.8,.8,.7,None],5)["threshold"]==.7
    assert calibrate_coverage([.9,None,None,None,None],5)["target_reachable"] is False
    assert abs(spatial_confidence(np.full((64,64),1./4096)))<1e-12
    source_check=None
    if vendor_metrics:
        import importlib.util
        spec=importlib.util.spec_from_file_location("released_native_evaluation",str(vendor_metrics))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        pred=np.asarray([[.1,.4],[.2,.3]],dtype=np.float64)
        gt=np.asarray([[.4,.1],[.3,.2]],dtype=np.float64)
        actual=native_heatmap_metrics(pred,gt)
        differences={key:abs(actual[key]-getattr(module,function)(pred,gt)) for key,function in
                     (("KLD","cal_kl"),("SIM","cal_sim"),("NSS","cal_nss"))}
        assert max(differences.values())<1e-12
        source_check={"source_sha256":sha256(vendor_metrics),"absolute_differences":differences}
    print(json.dumps({"status":"PASS","checks":["author minmax including negative logits and constant maps",
        "native metrics hand example","crop projection and mass; DT/tie geometry",
        "coverage thresholds and undefined-score denominator; entropy"],"native_source_check":source_check}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-head",action="store_true")
    parser.add_argument("--score",action="store_true")
    parser.add_argument("--root",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--calibration-manifest",type=Path)
    parser.add_argument("--predictions",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--additional-predictions",type=Path)
    parser.add_argument("--sealed-head",type=Path)
    parser.add_argument("--vendor-metrics",type=Path)
    parser.add_argument("--plan",type=Path,default=Path(__file__).with_name("PLAN.md"))
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test(args.vendor_metrics);return
    if args.select_head:
        if args.calibration_manifest is None or args.predictions is None or args.output_dir is None:
            parser.error("--select-head needs --calibration-manifest --predictions --output-dir")
        select_head(args);return
    if args.score:
        if args.root is None:parser.error("--score needs --root")
        evaluate(args);return
    parser.error("Use --select-head, --score, or --self-test")


if __name__=="__main__":
    main()
