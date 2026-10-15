"""PerSAM (Zhang et al., ICLR 2024) identification on a fixed ORBIT frame subset, adapted as in Swiss DINO (Sec. V-D1):
SAM ViT-B features averaged over the support mask give a prototype per object; for a query frame PerSAM's
positive/negative location prior (argmax/argmin cosine to the target embedding) prompts the SAM decoder, and the class
score is the cosine between the mean query feature inside the predicted mask and the prototype. EXPLORE, 2026-10-15.

No annotations: the support mask is SAM's best multimask output for a single centre click on each context frame.
SAM is ~60x the per-frame compute of DINOv2-S, so it runs on `per_video` random valid frames per clutter video (seed 1);
the DINOv2 arms from an orbit_dump directory are re-scored on exactly the same frames (single frame, h0).
PerSAM's target-guided attention / target-semantic prompting (segmentation-quality add-ons) are not included.
Usage: persam_subset.py <split dir> <dump dirs, comma-separated> <out json> [per_video=10] [support_per_object=8] [users]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orbit_eval import read_many  # noqa: E402
from orbit_dump import guard, peaks  # noqa: E402


def fnum(f):
    return int(f.rsplit('-', 1)[1].split('.')[0])


def main(root, dump, out, per_video=10, support_per_object=8, only_users=''):
    import torch
    import torch.nn.functional as F
    from transformers import SamModel, SamProcessor
    guard()
    root = Path(root); dumps = [Path(d) for d in dump.split(',')]; per_video, spo = int(per_video), int(support_per_object)
    plan = json.loads((root / 'plan.json').read_text())
    model = SamModel.from_pretrained('facebook/sam-vit-base').cuda().eval().half()
    proc = SamProcessor.from_pretrained('facebook/sam-vit-base')
    S = 1024 / 224.

    def embed(x):                                                   # uint8 [N,224,224,3] -> [N,256,64,64]
        out = []
        for k in range(0, len(x), 2):                              # SAM's global-attention rel-pos bias is ~0.4 GB/image
            pv = proc(images=list(x[k:k + 2]), return_tensors='pt')['pixel_values'].cuda().half()
            with torch.inference_mode():
                out.append(model.get_image_embeddings(pv).float())
        return torch.cat(out)

    def decode(emb, pts, labels):                                   # emb [1,256,64,64]; pts [P,n,2] (1024 px)
        with torch.inference_mode():
            o = model(image_embeddings=emb.half(), input_points=pts[None].half(), input_labels=labels[None],
                      multimask_output=True)
        best = o.iou_scores[0].argmax(-1)                           # [P]
        m = o.pred_masks[0][torch.arange(len(best)), best]          # [P,256,256] logits
        return F.interpolate(m[:, None].float(), (64, 64), mode='bilinear')[:, 0] > 0

    users = {}
    for v in plan['videos']:
        users.setdefault(v['user'], []).append(v)
    rng = random.Random(1); rows, t0 = [], time.time()
    for user, vids in sorted(users.items()):
        if only_users and user not in only_users.split(','):
            continue
        objs = sorted({v['obj'] for v in vids if v['type'] == 'clean'})
        protos, allsup = [], []
        for o in objs:
            ctx = [f for v in vids if v['type'] == 'clean' and v['obj'] == o for f in v['context'] if fnum(f) <= 1000]
            ctx = [ctx[i] for i in np.linspace(0, len(ctx) - 1, min(spo, len(ctx))).round().astype(int)]
            e = embed(read_many(root / 'frames', ctx))
            feats = []
            for i in range(len(e)):
                mk = decode(e[i:i + 1], torch.tensor([[[512., 512.]]], device='cuda'), torch.ones(1, 1, dtype=torch.long, device='cuda'))[0]
                if mk.sum() == 0:
                    mk = torch.ones_like(mk)
                feats.append(e[i][:, mk].mean(1))
            protos.append(torch.stack(feats).mean(0)); allsup.append(e.flatten(2).mean(2).mean(0))
        mu = torch.stack(allsup).mean(0)                               # user-wide mean SAM feature (support only)
        Pc = F.normalize(torch.stack(protos) - mu, dim=1)              # centred prototypes (diagnostic variant)
        P = F.normalize(torch.stack(protos), dim=1)                     # [O,256] raw-cosine prototypes (PerSAM / Swiss DINO adaptation)
        for v in vids:
            if v['type'] != 'clutter':
                continue
            valid = [f for f in v['valid'] if fnum(f) <= 1000]
            if len(valid) < 50:
                continue
            pick = sorted(rng.sample(valid, min(per_video, len(valid))))
            e = embed(read_many(root / 'frames', pick))
            ef = F.normalize(e.flatten(2), dim=1)                       # [N,256,4096]
            scores, s_max, s_glob, s_cent = [], [], [], []
            for i in range(len(e)):
                sim = P @ ef[i]                                         # [O,4096]
                s_max.append(sim.amax(1)); s_glob.append(F.normalize(e[i].flatten(1).mean(1), dim=0) @ P.T)
                pos, neg = sim.argmax(1), sim.argmin(1)
                xy = lambda idx: torch.stack([(idx % 64).float() * 16 + 8, (idx // 64).float() * 16 + 8], 1)
                pts = torch.stack([xy(pos), xy(neg)], 1)                # [O,2,2]
                lab = torch.tensor([[1, 0]] * len(objs), device='cuda')
                masks = decode(e[i:i + 1], pts, lab)                    # [O,64,64]
                fm = e[i].flatten(1)                                    # [256,4096]
                mf = torch.stack([fm[:, mk.flatten()].mean(1) if mk.any() else fm[:, pos[j]] for j, mk in enumerate(masks)])
                scores.append((F.normalize(mf, dim=1) * P).sum(1)); s_cent.append((F.normalize(mf - mu, dim=1) * Pc).sum(1))
            s = torch.stack(scores).cpu().numpy()
            s_max = torch.stack(s_max).cpu().numpy(); s_glob = torch.stack(s_glob).cpu().numpy(); s_cent = torch.stack(s_cent).cpu().numpy()
            y = objs.index(v['obj'])
            row = dict(video=v['video'], user=user, persam=float((s.argmax(1) == y).mean()),
                       diag_sam_maxsim=float((s_max.argmax(1) == y).mean()), diag_sam_global=float((s_glob.argmax(1) == y).mean()),
                       persam_centred=float((s_cent.argmax(1) == y).mean()))
            for dd in dumps:
                d = np.load(dd / f"{v['video']}.npz")
                pos_in = {n: k for k, n in enumerate(d['names'].tolist())}
                idx = [pos_in[f] for f in pick]
                for a in [k for k in d.files if k not in ('names', 'valid', 'y', 'user')]:
                    row[f'{dd.name}:{a}'] = float((d[a][idx].argmax(1) == y).mean())
            rows.append(row)
        print(user, len(objs), 'objects', round(time.time() - t0), 's', json.dumps(peaks()), flush=True)
        torch.cuda.empty_cache(); guard()
    arms = [k for k in rows[0] if k not in ('video', 'user')]
    res = dict(per_video=per_video, support_per_object=spo, n_videos=len(rows), wall_s=round(time.time() - t0, 1),
               acc={a: [round(100 * float(np.mean([r[a] for r in rows])), 2),
                        round(100 * 1.96 * float(np.std([r[a] for r in rows])) / np.sqrt(len(rows)), 2)] for a in arms})
    Path(out).write_text(json.dumps(dict(res, rows=rows), indent=1))
    print(json.dumps(res['acc']), res['wall_s'], 's', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
