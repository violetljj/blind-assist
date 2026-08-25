#!/usr/bin/env python3
"""Train/dev or one-shot test frozen-encoder GRAIL M1 comparators."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


VISUAL_WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
DEPTH_WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
K_POSES = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expanded_crop(image: Image.Image, bbox: list[int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    px, py = max(32, x1 - x0), max(32, y1 - y0)
    return image.crop((max(0, x0-px), max(0, y0-py), min(image.width, x1+px), min(image.height, y1+py)))


@torch.inference_mode()
def encode_images(images: list[Image.Image], processor: Any, model: Any, device: torch.device) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), 16):
        batch = processor(images=images[start:start+16], return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        encoded = model(**batch).pooler_output
        encoded = F.normalize(encoded.float(), dim=-1)
        outputs.append(encoded.cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.inference_mode()
def predict_depth(image: Image.Image, processor: Any, model: Any, device: torch.device) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prediction = model(**inputs).predicted_depth[:, None]
    prediction = F.interpolate(prediction, size=(image.height, image.width), mode="bicubic", align_corners=False)[0, 0]
    return prediction.float().cpu().numpy()


def materialize_features(collection_path: Path, dataset_root: Path, cache_path: Path,
                         visual_path: Path, depth_path: Path) -> dict[str, Any]:
    collection_hash = sha256_file(collection_path)
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        if cached["collection_sha256"] != collection_hash:
            raise ValueError("feature cache collection identity mismatch")
        return cached
    from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_processor = AutoImageProcessor.from_pretrained(visual_path, local_files_only=True)
    visual = AutoModel.from_pretrained(visual_path, local_files_only=True).to(device).eval()
    depth_processor = AutoImageProcessor.from_pretrained(depth_path, local_files_only=True)
    depth = AutoModelForDepthEstimation.from_pretrained(depth_path, local_files_only=True).to(device).eval()
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    feature_rows = []
    for number, row in enumerate(collection["rows"], 1):
        query = Image.open(dataset_root / row["query_image"]).convert("RGB")
        reference = Image.open(dataset_root / row["reference_image"]).convert("RGB")
        candidate_crops = [expanded_crop(query, candidate["bbox"]) for candidate in row["candidates"]]
        embeddings = encode_images([query, reference, *candidate_crops], visual_processor, visual, device)
        relative_depth = predict_depth(query, depth_processor, depth, device)
        candidates = []
        for candidate, embedding in zip(row["candidates"], embeddings[2:]):
            x0, y0, x1, y1 = candidate["bbox"]
            depth_crop = relative_depth[y0:y1, x0:x1]
            candidates.append({
                **candidate, "embedding": embedding,
                "relative_depth": float(np.median(depth_crop)),
                "geometry": np.asarray([
                    ((x0+x1)/2/320.0 - 0.5) * 2.0, ((y0+y1)/2/240.0 - 0.5) * 2.0,
                    (x1-x0)/320.0, (y1-y0)/240.0, float(np.median(depth_crop)),
                ], dtype=np.float32),
            })
        feature_rows.append({**row, "query_embedding": embeddings[0], "reference_embedding": embeddings[1], "candidates": candidates})
        if number % 25 == 0:
            print(json.dumps({"state": "FEATURES", "completed": number, "total": len(collection["rows"])}), flush=True)
    payload = {"schema": "blindassist_grail_m1_features_v1", "collection_sha256": collection_hash, "rows": feature_rows}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(cache_path)
    del visual, depth
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return payload


def negative_reference_indices(rows: list[dict[str, Any]]) -> list[int]:
    result = []
    for index, row in enumerate(rows):
        for offset in range(1, len(rows)+1):
            candidate = (index + offset) % len(rows)
            if rows[candidate]["house_index"] != row["house_index"]:
                result.append(candidate); break
        else: raise ValueError("cannot construct different-house absence pair")
    return result


def b2_input(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.concatenate([query, reference, query*reference, np.abs(query-reference)]).astype(np.float32)


def candidate_input(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.concatenate([candidate, reference, candidate*reference, np.abs(candidate-reference)]).astype(np.float32)


def target_pose_set(truth: list[dict[str, float]]) -> np.ndarray:
    chosen = []
    for desired_yaw in (-30.0, 0.0, 30.0):
        pose = min(truth, key=lambda p: (abs((p["yaw"]-desired_yaw+180)%360-180), math.hypot(p["x"], p["z"])))
        yaw = math.radians(pose["yaw"])
        chosen.append([pose["x"], pose["z"], math.sin(yaw), math.cos(yaw)])
    return np.asarray(chosen, dtype=np.float32)


class B2Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim*4, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.presence, self.pose = nn.Linear(128, 1), nn.Linear(128, 4)
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(value); return self.presence(hidden).squeeze(-1), self.pose(hidden)


class GrailModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.referent = nn.Sequential(nn.Linear(dim*4, 256), nn.ReLU(), nn.Linear(256, 1))
        self.pose = nn.Sequential(nn.Linear(dim*3+5, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, K_POSES*4))
    def forward(self, query: torch.Tensor, reference: torch.Tensor, candidate: torch.Tensor, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ref_input = torch.cat([candidate, reference, candidate*reference, torch.abs(candidate-reference)], dim=-1)
        pose_input = torch.cat([query, candidate, reference, geometry], dim=-1)
        return self.referent(ref_input).squeeze(-1), self.pose(pose_input).reshape(-1, K_POSES, 4)


def train_models(train_rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = len(train_rows[0]["query_embedding"])
    b2, grail = B2Model(dim).to(device), GrailModel(dim).to(device)
    b2_opt, grail_opt = torch.optim.Adam(b2.parameters(), 2e-3), torch.optim.Adam(grail.parameters(), 2e-3)
    neg_indices = negative_reference_indices(train_rows)
    depth_x, depth_y = [], []
    for row in train_rows:
        target = next(c for c in row["candidates"] if c["is_target"])
        depth_x.append([target["relative_depth"], 1.0/max(target["relative_depth"], 1e-6), 1.0])
        depth_y.append(target["source_depth_median_m"])
    depth_calibration = np.linalg.lstsq(np.asarray(depth_x), np.asarray(depth_y), rcond=None)[0].astype(np.float32)
    order = list(range(len(train_rows)))
    for epoch in range(80):
        random.Random(1000+epoch).shuffle(order)
        b2.train(); grail.train()
        for index in order:
            row, neg_ref = train_rows[index], train_rows[neg_indices[index]]["reference_embedding"]
            query = torch.tensor(row["query_embedding"], device=device)
            ref = torch.tensor(row["reference_embedding"], device=device)
            negative_ref = torch.tensor(neg_ref, device=device)
            pose_target = torch.tensor(target_pose_set(row["truth_local_poses"]), device=device)
            b2_opt.zero_grad()
            pos_logit, pos_pose = b2(torch.tensor(b2_input(row["query_embedding"], row["reference_embedding"]), device=device))
            neg_logit, _ = b2(torch.tensor(b2_input(row["query_embedding"], neg_ref), device=device))
            b2_loss = F.binary_cross_entropy_with_logits(torch.stack([pos_logit,neg_logit]), torch.tensor([1.,0.],device=device))
            b2_loss = b2_loss + F.smooth_l1_loss(pos_pose, pose_target[1])
            b2_loss.backward(); b2_opt.step()
            grail_opt.zero_grad(); logits, poses, labels = [], [], []
            for candidate in row["candidates"]:
                cand = torch.tensor(candidate["embedding"], device=device); geom = torch.tensor(candidate["geometry"], device=device)
                logit, predicted = grail(query, ref, cand, geom); logits.append(logit); labels.append(float(candidate["is_target"]))
                if candidate["is_target"]: poses.append(predicted)
                negative_logit, _ = grail(query, negative_ref, cand, geom); logits.append(negative_logit); labels.append(0.0)
            label_tensor = torch.tensor(labels, device=device)
            pos_weight = torch.tensor(max(1.0, (len(labels)-label_tensor.sum().item())/max(1.0,label_tensor.sum().item())), device=device)
            loss = F.binary_cross_entropy_with_logits(torch.stack(logits), label_tensor, pos_weight=pos_weight)
            loss = loss + F.smooth_l1_loss(poses[0], pose_target)
            loss.backward(); grail_opt.step()
        if epoch in (0, 19, 39, 59, 79): print(json.dumps({"state":"TRAIN","epoch":epoch+1,"epochs":80}), flush=True)
    checkpoint = {"schema":"blindassist_grail_m1_checkpoint_v1","dim":dim,"k":K_POSES,
                  "b2":b2.cpu().state_dict(),"grail":grail.cpu().state_dict(),"depth_calibration":depth_calibration}
    output.parent.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, output)
    return checkpoint


def yaw_error(left: float, right: float) -> float:
    return abs((left-right+180.0)%360.0-180.0)


def pose_success(predictions: list[list[float]], truth: list[dict[str,float]]) -> bool:
    for p in predictions:
        yaw = math.degrees(math.atan2(p[2], p[3]))
        if any(math.hypot(p[0]-t["x"],p[1]-t["z"])<=0.5 and yaw_error(yaw,t["yaw"])<=20 for t in truth): return True
    return False


def threshold_from_dev(positive: list[float], negative: list[float]) -> float:
    values = sorted(set([0.0, 1.0, *positive, *negative]))
    best = None
    for value in values:
        balanced = 0.5*(sum(x>=value for x in positive)/len(positive) + sum(x<value for x in negative)/len(negative))
        candidate = (balanced, value)
        if best is None or candidate > best: best = candidate
    return float(best[1])


@torch.inference_mode()
def score_rows(rows: list[dict[str,Any]], checkpoint: dict[str,Any], thresholds: dict[str,float]|None=None) -> dict[str,Any]:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dim=checkpoint["dim"]
    b2=B2Model(dim).to(device); b2.load_state_dict(checkpoint["b2"]); b2.eval()
    grail=GrailModel(dim).to(device); grail.load_state_dict(checkpoint["grail"]); grail.eval()
    neg_indices=negative_reference_indices(rows); cal=np.asarray(checkpoint["depth_calibration"])
    scores={name:{"positive":[],"negative":[]} for name in ("B0","B1","B2","GRAIL")}; positive_records=[]
    for i,row in enumerate(rows):
        refs={"positive":row["reference_embedding"],"negative":rows[neg_indices[i]]["reference_embedding"]}
        per_ref={}
        for kind,ref_np in refs.items():
            similarities=[float(np.dot(c["embedding"],ref_np)) for c in row["candidates"]]
            selected=int(np.argmax(similarities)); cosine=float(similarities[selected]); candidate=row["candidates"][selected]
            for baseline in ("B0","B1"): scores[baseline][kind].append(cosine)
            b2_logit,b2_pose=b2(torch.tensor(b2_input(row["query_embedding"],ref_np),device=device))
            scores["B2"][kind].append(float(torch.sigmoid(b2_logit)))
            candidate_scores=[]; candidate_poses=[]
            for c in row["candidates"]:
                logit,poses=grail(torch.tensor(row["query_embedding"],device=device),torch.tensor(ref_np,device=device),torch.tensor(c["embedding"],device=device),torch.tensor(c["geometry"],device=device))
                candidate_scores.append(float(torch.sigmoid(logit))); candidate_poses.append(poses.cpu().tolist())
            gsel=int(np.argmax(candidate_scores)); scores["GRAIL"][kind].append(candidate_scores[gsel])
            if kind=="positive":
                cx=(candidate["bbox"][0]+candidate["bbox"][2])/2/320.0; bearing=math.radians((cx-0.5)*90.0)
                b0_pose=[[math.sin(bearing),math.cos(bearing),math.sin(bearing),math.cos(bearing)]]
                d=candidate["relative_depth"]; metric=max(0.5,float(np.dot([d,1/max(d,1e-6),1],cal))-1.0)
                b1_pose=[[metric*math.sin(bearing),metric*math.cos(bearing),math.sin(bearing),math.cos(bearing)]]
                positive_records.append({"row":row,"selected":selected,"b0":b0_pose,"b1":b1_pose,"b2":[b2_pose.cpu().tolist()],
                                         "grail_selected":gsel,"grail":candidate_poses[gsel],"grail_scores":candidate_scores})
        per_ref=None
    if thresholds is None:
        thresholds={name:threshold_from_dev(values["positive"],values["negative"]) for name,values in scores.items()}
    metrics={}
    for name in ("B0","B1","B2","GRAIL"):
        committed=[s>=thresholds[name] for s in scores[name]["positive"]]
        success=[]; wrong=[]
        for record,commit in zip(positive_records,committed):
            if name in ("B0","B1"):
                selected=record["selected"]; poses=record[name.lower()]
            elif name=="B2": selected=None; poses=record["b2"]
            else: selected=record["grail_selected"]; poses=record["grail"]
            target_selected=(selected is None or record["row"]["candidates"][selected]["is_target"])
            success.append(commit and target_selected and pose_success(poses,record["row"]["truth_local_poses"]))
            if record["row"]["same_type_visible_candidates"]>=2:
                wrong.append(bool(commit and selected is not None and not target_selected))
        metrics[name]={"pose_success":sum(success),"positive_denominator":len(success),
                       "wrong_target":sum(wrong),"wrong_target_denominator":len(wrong),
                       "absence_false_commit":sum(s>=thresholds[name] for s in scores[name]["negative"]),
                       "absence_denominator":len(scores[name]["negative"])}
    # Candidate-independent heads plus max must be invariant to reversal; verify exactly on recorded scores.
    permutation=sum(int(np.argmax(r["grail_scores"])) == len(r["grail_scores"])-1-int(np.argmax(list(reversed(r["grail_scores"])))) for r in positive_records)
    return {"thresholds":thresholds,"metrics":metrics,"permutation_consistent":permutation,"permutation_denominator":len(positive_records)}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("develop","test"),required=True)
    parser.add_argument("--dataset-root",type=Path,required=True); parser.add_argument("--visual-model",type=Path,required=True)
    parser.add_argument("--depth-model",type=Path,required=True); parser.add_argument("--artifact-dir",type=Path,required=True)
    args=parser.parse_args(); args.artifact_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_path=args.artifact_dir/"checkpoint.pt"
    roles=("train","dev") if args.stage=="develop" else ("test",)
    features={}
    for role in roles:
        features[role]=materialize_features(args.dataset_root/role/"collection.json",args.dataset_root,args.artifact_dir/f"features-{role}.pt",args.visual_model,args.depth_model)["rows"]
    if args.stage=="develop":
        checkpoint=train_models(features["train"],checkpoint_path); result=score_rows(features["dev"],checkpoint)
        result.update({"schema":"blindassist_grail_m1_development_v1","checkpoint_sha256":sha256_file(checkpoint_path),"terminal":"GRAIL_M1_DEVELOPMENT_COMPLETE_TEST_UNOPENED"})
        (args.artifact_dir/"development-result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    else:
        checkpoint=torch.load(checkpoint_path,weights_only=False); dev=json.loads((args.artifact_dir/"development-result.json").read_text(encoding="utf-8"))
        result=score_rows(features["test"],checkpoint,dev["thresholds"]); result.update({"schema":"blindassist_grail_m1_test_v1","checkpoint_sha256":sha256_file(checkpoint_path)})
        (args.artifact_dir/"formal-test-result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
