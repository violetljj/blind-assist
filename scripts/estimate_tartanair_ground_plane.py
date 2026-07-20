#!/usr/bin/env python3
"""GPU RANSAC audit of lower-image candidate ground planes in an extracted TartanAir slice."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def estimate(depth: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray, torch):
    device = torch.device("cuda")
    v, u = torch.meshgrid(torch.arange(280, depth.shape[0], 8, device=device), torch.arange(160, 480, 8, device=device), indexing="ij")
    d = torch.as_tensor(depth, device=device)[v, u]
    points = torch.stack(((u-intrinsics[0,2])*d/intrinsics[0,0], (v-intrinsics[1,2])*d/intrinsics[1,1], d), -1)
    pose_t = torch.as_tensor(pose, device=device); world = (points @ pose_t[:3,:3].T + pose_t[:3,3]).reshape(-1, 3)
    count = world.shape[0]; generator=torch.Generator(device=device).manual_seed(20260720)
    triples = world[torch.randint(count, (256,3), generator=generator, device=device)]
    normals = torch.linalg.cross(triples[:,1]-triples[:,0], triples[:,2]-triples[:,0]); normals = normals / torch.linalg.vector_norm(normals,dim=1,keepdim=True).clamp_min(1e-6)
    offsets = -(normals*triples[:,0]).sum(1)
    distances = torch.abs(world @ normals.T + offsets)
    scores = (distances < .08).sum(0); best=int(scores.argmax())
    inliers = world[distances[:,best] < .08]; centroid=inliers.mean(0); _,_,vh=torch.linalg.svd(inliers-centroid); normal=vh[-1]; offset=-(normal*centroid).sum()
    camera=pose_t[:3,3]; height=torch.abs(normal@camera+offset)
    return {"inlier_fraction":float(inliers.shape[0]/count),"camera_plane_distance_m":float(height.cpu()),"normal_world":normal.cpu().tolist(),"offset_world":float(offset.cpu()),"sample_count":int(count)}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True); args=parser.parse_args(); import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    manifest=json.loads((args.root/"slice_manifest.json").read_text(encoding="utf-8").strip().removesuffix("\\n")); trajectory=args.root/manifest["trajectory"]; rows=[]
    for ident in manifest["frame_ids"]:
        cam=np.load(trajectory/f"{ident}_cam.npz"); rows.append({"frame_id":ident,**estimate(np.load(trajectory/f"{ident}_depth.npy"),cam["camera_pose"],cam["camera_intrinsics"],torch)})
    inliers=[r["inlier_fraction"] for r in rows]; heights=[r["camera_plane_distance_m"] for r in rows]
    median_inlier=float(np.median(inliers)); median_height=float(np.median(heights)); height_span=float(max(heights)-min(heights))
    geometric_gate = median_inlier >= .60 and .80 <= median_height <= 2.20 and height_span <= .50
    report={"format":"blindassist_tartanair_ground_plane_candidate_audit_v1","frame_count":len(rows),"cuda_device":torch.cuda.get_device_name(0),"frames":rows,"median_inlier_fraction":median_inlier,"median_camera_plane_distance_m":median_height,"camera_plane_distance_span_m":height_span,"geometric_ground_candidate_passed":geometric_gate,"geometric_gate":{"minimum_median_inlier_fraction":.60,"camera_height_range_m":[.80,2.20],"maximum_height_span_m":.50},"ustrf_ground_plane_admitted":False,"reason":"candidate RANSAC plane has no independent world-to-body/up-axis verification"}
    qa=args.root/"qa"; qa.mkdir(exist_ok=True); (qa/"ground_plane_candidate_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"frames":len(rows),"median_inlier":report["median_inlier_fraction"],"median_height":report["median_camera_plane_distance_m"]}))
if __name__=="__main__": main()
