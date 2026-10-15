"""EXPLORE only: fixed consumed SANPO frames, no evaluator mask in selectors.

Run from the BlindAssist checkout with its existing torch/transformers runtime.
No network, fitting, threshold search, protected inputs, or production writes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import SegformerForSemanticSegmentation


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_detail_selector(rgb):
    """Select two fixed quadrants from RGB-only vertical-detail residual energy."""
    gray = cv2.cvtColor(cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    coarse = cv2.resize(cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA), (512, 512), interpolation=cv2.INTER_LINEAR)
    residual = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) - cv2.Sobel(coarse, cv2.CV_32F, 1, 0, ksize=3))
    scores = [float(np.mean(a)) for a in split_four(residual)]
    return sorted(range(4), key=lambda i: (-scores[i], i))[:2], scores


def split_four(a):
    h, w = a.shape[:2]
    return [a[:h//2, :w//2], a[:h//2, w//2:], a[h//2:, :w//2], a[h//2:, w//2:]]


def bounds(shape, q):
    h, w = shape[:2]
    return (0 if q < 2 else h//2, h//2 if q < 2 else h,
            0 if q % 2 == 0 else w//2, w//2 if q % 2 == 0 else w)


def metrics(pred, truth, valid):
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth & valid))
    fn = int(np.count_nonzero(~pred & truth))
    return dict(tp=tp, fp=fp, fn=fn, gt_pixels=int(truth.sum()), valid_pixels=int(valid.sum()),
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                iou=tp/(tp+fp+fn) if tp+fp+fn else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts.local/work/assistive-thesis-screen-20261015/thin_structure/run1")
    ap.add_argument("--max-wall", type=float, default=280)
    ap.add_argument("--model-dir")
    ap.add_argument("--target-ids", default="87,93")
    ap.add_argument("--strong1024", action="store_true")
    args = ap.parse_args()
    start = time.perf_counter()
    root = Path.cwd()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions").mkdir()
    (out / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    source = root / "artifacts.local/work/data-potential-dev-20261013/round4/sanpo_provenance/catalog.json"
    model_dir = root / (args.model_dir or "artifacts.local/models/nvidia--segformer-b0-finetuned-ade-512-512/snapshots/489d5cd81a0b59fab9b7ea758d3548ebe99677da")
    target_ids = [int(x) for x in args.target_ids.split(',')]
    weight = model_dir/("model.safetensors" if (model_dir/"model.safetensors").exists() else "pytorch_model.bin")
    rows = json.loads(source.read_text(encoding="utf8"))["frames"]
    assert len(rows) == 50
    assert all(r["split"] == "official_train" and r["source_role"] == "already_consumed_Development" and r["annotation_type"] == "HUMAN_ANNOTATED" for r in rows)
    manifest = {"phase":"EXPLORE", "selection":"all 50 rows in existing human provenance catalogue; no new sample selection",
        "catalogue":str(source), "catalogue_sha256":sha(source), "code_sha256":sha(Path(__file__)),
        "budgets":{"cpu_command_wall_s":600, "gpu_process_wall_s":300,"downloads_bytes":300000000},
        "adjustable_scope":"stop before starting another frame if max wall reached; preserve completed and missing frames",
        "deliverable":"paired model/representation metrics and selector audit", "samples":[{k:r[k] for k in ["session","camera","frame","split","source_role","annotation_type","assets"]} for r in rows],
        "model_dir":str(model_dir), "model_sha256":sha(weight),
        "target_mapping":{"SANPO":[24],"model_ids":target_ids,"note":"ADE streetlight/pole or Cityscapes pole; imperfect cross-dataset class correspondence"},
        "methods":{"low256":"whole-image 256x256", "full512":"whole-image 512x512 reference", "entropy2":"low256 + highest mean entropy two quadrants, each 256x256", "rgbdetail2":"low256 + highest RGB vertical-detail residual two quadrants, each 256x256"},
        "fixed_decisions":{"selected_quadrants":2,"patch_fusion":"replace each selected quadrant with its patch-only prediction","instance_recall_threshold":0.2,"mask_representation_threshold":0.5},
        "claim_boundary":"cross-domain pretrained model pilot; no SANPO-trained strong baseline; no distance, safety, device latency or novel-method claim"}
    save_json(out/"manifest.json", manifest)
    manifest["strong1024"] = args.strong1024
    if args.strong1024:
        manifest["methods"]["full1024"] = "whole-image 1024x1024 at Cityscapes checkpoint nominal resolution"
    save_json(out/"manifest.json", manifest)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True).eval()
    mean = torch.tensor([0.485,0.456,0.406])[None,:,None,None]
    std = torch.tensor([0.229,0.224,0.225])[None,:,None,None]

    def tensor(rgb, size, device):
        # Matches model preprocessor's PIL BILINEAR resize, rescale, normalize.
        small = np.asarray(Image.fromarray(rgb).resize((size,size), Image.Resampling.BILINEAR)).copy()
        x = torch.from_numpy(small).permute(2,0,1)[None].float()/255
        return ((x-mean)/std).to(device)

    first = np.asarray(Image.open(rows[0]["assets"]["video_frames"]["path"]).convert("RGB"))
    benchmark = {}
    with torch.inference_mode():
        for backend in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            model.to(backend)
            x = tensor(first,512,backend)
            model(pixel_values=x)
            if backend == "cuda": torch.cuda.synchronize()
            times=[]
            for _ in range(3):
                t=time.perf_counter(); model(pixel_values=x)
                if backend == "cuda": torch.cuda.synchronize()
                times.append(time.perf_counter()-t)
            benchmark[backend] = {"warm_forward_s":times,"median_s":float(np.median(times))}
        device = min(benchmark, key=lambda d:benchmark[d]["median_s"])
        model.to(device)
        benchmark["selected"] = device
        benchmark["device_name"] = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        benchmark["torch_threads"] = torch.get_num_threads()
        benchmark["inference_includes"] = "model forward only; excludes decode, resize, logits upsampling, CPU copies and I/O"
        save_json(out/"backend_benchmark.json", benchmark)
        print(json.dumps(benchmark),flush=True)

        def infer(rgb,size):
            x=tensor(rgb,size,device)
            if device=="cuda": torch.cuda.synchronize()
            t=time.perf_counter()
            logits=model(pixel_values=x).logits
            if device=="cuda": torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            return logits, elapsed

        def decode(logits,shape):
            # Native-resolution argmax of bilinearly upsampled logits.
            labels=F.interpolate(logits,size=shape[:2],mode="bilinear",align_corners=False).argmax(1)[0]
            target=torch.zeros_like(labels,dtype=torch.bool)
            for ident in target_ids: target |= labels==ident
            return target.cpu().numpy()

        result_rows=[]; instances=[]; selector_rows=[]; visual=[]; failures=[]
        for i,r in enumerate(rows):
            if time.perf_counter()-start > args.max_wall:
                failures.append({"kind":"BUDGET_STOP","unprocessed_samples":len(rows)-i}); break
            t_frame=time.perf_counter()
            try:
                # Hash identity before pixel read. GT is loaded only after candidate generation.
                for key in ["video_frames","segmentation_masks"]:
                    assert sha(r["assets"][key]["path"])==r["assets"][key]["sha256"]
                rgb=np.asarray(Image.open(r["assets"]["video_frames"]["path"]).convert("RGB"))
                low_logits,low_s=infer(rgb,256)
                probs=low_logits.softmax(1)
                entropy=(-(probs*probs.clamp_min(1e-9).log()).sum(1)[0]).cpu().numpy()
                entropy_scores=[float(np.mean(a)) for a in split_four(entropy)]
                entropy_q=sorted(range(4),key=lambda q:(-entropy_scores[q],q))[:2]
                rgb_q,rgb_scores=rgb_detail_selector(rgb)
                low=decode(low_logits,rgb.shape)
                full_logits,full_s=infer(rgb,512)
                full=decode(full_logits,rgb.shape)
                del full_logits,low_logits,probs
                patch_masks={}; patch_s={}
                for q in sorted(set(entropy_q+rgb_q)):
                    y0,y1,x0,x1=bounds(rgb.shape,q)
                    crop=rgb[y0:y1,x0:x1]
                    logits,elapsed=infer(crop,256)
                    patch_masks[q]=decode(logits,crop.shape); patch_s[q]=elapsed
                    del logits
                predictions={"low256":low,"full512":full}
                high_s = None
                if args.strong1024:
                    high_logits,high_s=infer(rgb,1024)
                    predictions["full1024"]=decode(high_logits,rgb.shape)
                    del high_logits
                for name,qs in [("entropy2",entropy_q),("rgbdetail2",rgb_q)]:
                    fused=low.copy()
                    for q in qs:
                        y0,y1,x0,x1=bounds(rgb.shape,q); fused[y0:y1,x0:x1]=patch_masks[q]
                    predictions[name]=fused
                # Evaluator-only section: no truth is passed to infer or either selector.
                pan=np.asarray(Image.open(r["assets"]["segmentation_masks"]["path"]).convert("RGB"))
                truth=pan[:,:,0]==24
                instance=pan[:,:,1].astype(np.uint16)*256+pan[:,:,2]
                for size in [256,512]:
                    p=cv2.resize(truth.astype(np.float32),(size,size),interpolation=cv2.INTER_AREA)
                    predictions[f"oracle_mask_{size}"]=cv2.resize((p>=.5).astype(np.uint8),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
                identity={"sample_index":i,"session":r["session"],"camera":r["camera"],"frame":r["frame"]}
                for name,pred in predictions.items():
                    times={"low256":low_s,"full512":full_s,"full1024":high_s,"entropy2":low_s+sum(patch_s[q] for q in entropy_q),"rgbdetail2":low_s+sum(patch_s[q] for q in rgb_q)}
                    result_rows.append({**identity,"method":name,**metrics(pred,truth,pan[:,:,0]!=0),"model_forward_s":times.get(name),"input_pixels":{"low256":256**2,"full512":512**2,"full1024":1024**2,"entropy2":3*256**2,"rgbdetail2":3*256**2}.get(name)})
                    for ident in np.unique(instance[truth]):
                        target=truth & (instance==ident)
                        recall=float((pred&target).sum()/target.sum())
                        instances.append({**identity,"method":name,"instance":int(ident),"identity_known":int(ident)>0,"gt_pixels":int(target.sum()),"pixel_recall":recall,"any_hit":bool(recall>0),"recall_ge_20pct":bool(recall>=.2)})
                    if not name.startswith("oracle"):
                        Image.fromarray((pred*255).astype(np.uint8)).save(out/"predictions"/f"{i:02d}_{name}.png")
                for name,qs,scores in [("entropy2",entropy_q,entropy_scores),("rgbdetail2",rgb_q,rgb_scores)]:
                    selected=np.zeros(truth.shape,bool)
                    for q in qs:
                        y0,y1,x0,x1=bounds(rgb.shape,q);selected[y0:y1,x0:x1]=True
                    selector_rows.append({**identity,"selector":name,"quadrants":qs,"scores":scores,"selected_area_fraction":float(selected.mean()),"gt_pole_pixels_covered":int((truth&selected).sum()),"gt_pole_pixels":int(truth.sum())})
                if i%5==0:
                    panels=[]
                    for name,mask in [("native human pole",truth)]+[(n,predictions[n]) for n in ["low256","full512","entropy2","rgbdetail2"]]:
                        thumb=Image.fromarray(rgb).resize((320,180))
                        m=np.asarray(Image.fromarray(mask).resize((320,180),Image.Resampling.NEAREST))
                        a=np.asarray(thumb).copy();a[m]=(a[m]*.4+np.array([255,40,180])*.6).astype(np.uint8)
                        panel=Image.new("RGB",(320,204),"white");panel.paste(Image.fromarray(a),(0,24));ImageDraw.Draw(panel).text((4,4),f"{i:02d} {name}",fill="black");panels.append(panel)
                    sheet=Image.new("RGB",(1600,204));[sheet.paste(p,(j*320,0)) for j,p in enumerate(panels)];visual.append(sheet)
                print(json.dumps({"frame":i+1,"total":len(rows),"elapsed_s":round(time.perf_counter()-start,2),"frame_s":round(time.perf_counter()-t_frame,3)}),flush=True)
            except Exception as exc:
                failures.append({"sample_index":i,"exception":repr(exc)})
                print(json.dumps(failures[-1]),flush=True)
        for file,data in [("frame_metrics.csv",result_rows),("instance_metrics.csv",instances)]:
            if data:
                with (out/file).open("w",encoding="utf8",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        save_json(out/"selectors.json",selector_rows)
        save_json(out/"failures.json",failures)
        summary={"selected_frames":50,"completed_frames":len(result_rows)//(7 if args.strong1024 else 6),"sessions":len(set(r["session"] for r in rows)),"methods":{},"failures":failures,"elapsed_s":time.perf_counter()-start,"downloads_bytes":0,"strong_task_baseline":"NOT_RUN: no SANPO-trained model; B0 pretrained cross-domain reference only","backend":benchmark,"limits":manifest["claim_boundary"]}
        for name in sorted(set(r["method"] for r in result_rows)):
            mr=[r for r in result_rows if r["method"]==name];ir=[r for r in instances if r["method"]==name]
            tp,fp,fn=[sum(r[k] for r in mr) for k in ["tp","fp","fn"]]
            summary["methods"][name]={"frames":len(mr),"tp":tp,"fp":fp,"fn":fn,"iou":tp/(tp+fp+fn) if tp+fp+fn else None,"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,"instance_frame_groups":len(ir),"any_hit_groups":sum(r["any_hit"] for r in ir),"recall_ge_20pct_groups":sum(r["recall_ge_20pct"] for r in ir),"known_id_groups":sum(r["identity_known"] for r in ir),"forward_median_s":float(np.median([r["model_forward_s"] for r in mr])) if mr[0]["model_forward_s"] is not None else None}
        summary["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated() if device=="cuda" else 0
        save_json(out/"summary.json",summary)
        if visual:
            sheet=Image.new("RGB",(1600,204*len(visual)));[sheet.paste(p,(0,j*204)) for j,p in enumerate(visual)];sheet.save(out/"contact_sheet.jpg")
        print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
