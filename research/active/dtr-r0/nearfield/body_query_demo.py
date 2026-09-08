"""Build an offline, all-eval BODY/HEAD A/B single-frame viewer.

Only RGB is model input. Evaluator truth is read for explicitly separate display
and error summaries; it never changes the images or prediction overlays.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cache_file(root, entry, fallback):
    """Accept manifest paths with optional SHA256; keep reads in the cache."""
    entry = entry or fallback
    name = entry.get("path", fallback) if isinstance(entry, dict) else entry
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Cache path escapes its root: {name}")
    if isinstance(entry, dict) and entry.get("sha256"):
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
    return path


def png_url(array):
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def overlay(rgb, support):
    """Probability alpha, full-frame resize, no evaluator-derived cropping."""
    result = rgb.astype(np.float32)
    for plane, color in zip(support, ((42, 182, 197), (234, 151, 68))):
        full = np.asarray(Image.fromarray(plane.astype(np.float32)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
        alpha = .65 * full[..., None]
        result = result * (1 - alpha) + np.array(color) * alpha
    return png_url(np.clip(result, 0, 255).astype(np.uint8))


def validate_probabilities(array, shape, name, categorical=False):
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name}: expected finite {shape}, got {array.shape}")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must contain probabilities, not logits")
    if categorical and not np.allclose(array.sum(-1), 1, atol=1e-4):
        raise ValueError(f"{name} categorical probabilities must sum to one")


def uncertainty(near, thresholds):
    p = np.clip(np.asarray(near, dtype=float), 1e-7, 1 - 1e-7)
    t = np.clip(np.asarray(thresholds, dtype=float), 1e-7, 1 - 1e-7)
    return np.abs(np.log(p / (1 - p)) - np.log(t / (1 - t))) < .5


def build_payload(cache, run):
    cache, run = cache.resolve(), run.resolve()
    manifest = read_json(cache / "manifest.json")
    if manifest.get("status") != "PASS":
        raise ValueError("Only an accepted PASS cache may enter the demo")
    partitions = manifest["partitions"]
    part = partitions["eval"] if isinstance(partitions, dict) else next(
        p for p in partitions if p["role"] == "eval")
    evaluator = read_json(cache_file(cache, part.get("evaluator"), "evaluator/eval.json"))
    records = evaluator["records"]
    if len({r["sample_index"] for r in records}) != len(records):
        raise ValueError("Duplicate eval sample indices")
    rgb = np.load(cache_file(cache, part.get("rgb"), "model/eval/rgb.npy"), allow_pickle=False)
    n = len(records)
    if rgb.shape != (n, 144, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 eval RGB with shape N x 144 x 256 x 3")
    indices = part.get("sample_indices")
    if indices is not None and list(indices) != [r["sample_index"] for r in records]:
        raise ValueError("Eval records do not match manifest sample order")
    if evaluator.get("sample_indices", indices) != indices:
        raise ValueError("Evaluator sample indices do not match manifest")
    truth = {}
    for key in ("near", "counts"):
        truth[key] = np.load(cache_file(cache, evaluator.get(key),
            f"evaluator/eval_{key}.npy"), allow_pickle=False)
    if truth["near"].shape != (n, 2) or truth["counts"].shape != (n, 12):
        raise ValueError("Evaluator truth dimensions do not match eval RGB")
    if not np.isin(truth["near"], (-1, 0, 1)).all():
        raise ValueError("Expected near truth -1 (UNKNOWN), 0, or 1")
    if not np.isin(truth["counts"], (-1, 0, 1, 2, 3)).all():
        raise ValueError("Expected capped count truth -1 (UNKNOWN), 0, 1, 2, or 3")
    selection = read_json(run / "selection.json")
    predictions, summaries = {}, {}
    for arm in ("A", "B"):
        thresholds = np.asarray(selection[arm]["thresholds"], dtype=np.float64)
        if thresholds.shape != (2,) or not np.isfinite(thresholds).all() or np.any(thresholds < 0) or np.any(thresholds > np.nextafter(1.0, np.inf)):
            raise ValueError(f"Invalid {arm} per-head thresholds")
        with np.load(run / f"{arm}-eval.npz", allow_pickle=False) as data:
            pred = {k: data[k] for k in ("near", "support", "counts")}
        validate_probabilities(pred["near"], (n, 2), f"{arm}.near")
        validate_probabilities(pred["support"], (n, 2, 18, 32), f"{arm}.support")
        validate_probabilities(pred["counts"], (n, 12, 4), f"{arm}.counts", True)
        pred["thresholds"] = thresholds
        # Inclusive float64 comparison preserves the >1 all-negative sentinel.
        pred["positive"] = np.asarray(pred["near"], dtype=np.float64) >= thresholds
        pred["unknown"] = uncertainty(pred["near"], thresholds)
        summaries[arm] = []
        for h in range(2):
            y, p = truth["near"][:, h], pred["positive"][:, h]
            summaries[arm].append(dict(
                tp=int(((y == 1) & p).sum()), fp=int(((y == 0) & p).sum()),
                fn=int(((y == 1) & ~p).sum()), tn=int(((y == 0) & ~p).sum()),
                truthUnknown=int((y < 0).sum()), unknown=int(pred["unknown"][:, h].sum())))
        predictions[arm] = pred
    cases = []
    for i, record in enumerate(records):
        item = dict(record=record, rgb=png_url(rgb[i]),
                    truth={k: v[i].tolist() for k, v in truth.items()}, arms={})
        for arm, pred in predictions.items():
            item["arms"][arm] = dict(near=pred["near"][i].tolist(),
                counts=pred["counts"][i].tolist(), thresholds=pred["thresholds"].tolist(),
                positive=pred["positive"][i].tolist(), unknown=pred["unknown"][i].tolist(),
                overlay=overlay(rgb[i], pred["support"][i]))
        cases.append(item)
    if not cases:
        raise ValueError("No admitted eval cases")
    return dict(cases=cases, summaries=summaries, cache=str(cache), run=str(run))


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BODY / HEAD · 单帧 A/B Development</title>
<style>
:root{color-scheme:light;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#25313a;background:#eef1f3}
*{box-sizing:border-box}body{margin:0;padding:24px;max-width:1440px;margin:auto}h1{font-size:25px;margin:0 0 12px}h2{font-size:19px;margin:0 0 12px}p{line-height:1.65;margin:8px 0}.muted,small{color:#596975}header,section,article{background:white;border:1px solid #d6dfe3;border-radius:10px;padding:18px;margin-bottom:16px}.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{font-size:13px;display:grid;gap:5px}select,button{font:inherit;padding:9px;border:1px solid #becbd1;border-radius:6px;background:white;color:inherit}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.raw{display:grid;grid-template-columns:minmax(260px,1fr) 1fr;gap:22px}.frame{width:100%;display:block;background:#20272c;border-radius:5px;image-rendering:auto}.chip{display:inline-block;background:#edf2f4;padding:4px 8px;border-radius:4px;margin:4px 6px 4px 0}.body{border-left:4px solid #2ab6c5}.head{border-left:4px solid #ea9744}.unknown{color:#795921}.bad{color:#a63434}.good{color:#356246}table{width:100%;border-collapse:collapse;font-size:12px;margin:12px 0}td,th{text-align:left;padding:7px 4px;border-bottom:1px solid #e1e6e9;font-variant-numeric:tabular-nums}th{color:#52636f;font-weight:500}.summary{overflow:auto}summary{cursor:pointer;font-weight:600}#empty{padding:24px}a{color:#315c78}@media(max-width:850px){body{padding:12px}.grid,.raw{grid-template-columns:1fr}}
</style>
<header><h1>BODY / HEAD · 单帧 A/B 对照</h1>
<p>合成同世界 Development · 仅 eval 分区 · 每帧独立，无时序融合</p>
<p class="muted">模型只接收 RGB；固定相机标定与身体查询区域是共同先验。几何真值仅用于下方评估展示。此页保留所有已接纳 eval 样本，不代表真实街道泛化或安全性能。</p>
<details><summary>全量结果与来源</summary><div id="totals" class="summary"></div><p id="sources" class="muted"></p></details></header>
<section class="controls"><label>场景组<select id="group"></select></label><label>条件<select id="condition"></select></label><label>样本<select id="sample"></select></label><button id="prev">上一帧</button><button id="next">下一帧</button><span id="position" class="muted" aria-live="polite"></span></section>
<div id="empty" hidden>此组没有该条件，未替换为其他样本。</div>
<main id="main"><section class="raw"><div><h2>原始 RGB · 共享模型输入</h2><img id="rgb" class="frame" alt="完整单帧 RGB 模型输入"></div><div><h2 id="caseTitle"></h2><p id="caseInfo" class="muted"></p><p>两种方法使用同一帧，阈值来自保存的 selection.json，此页不能调阈值。</p><p><span class="chip body">青色 BODY</span><span class="chip head">橙色 HEAD</span></p><p class="muted">支持图叠加透明度 = 0.65 × 预测概率，完整帧最近邻上采样。颜色仅表达模型支持概率，不是几何真值或定位保证。</p><details><summary>评估器真值（非模型输入）</summary><p id="truthNear"></p><div id="truthCounts"></div></details></div></section>
<div class="grid"><article id="A"></article><article id="B"></article></div>
<section><p>主预测按各头 near 分数 ≥ 保存阈值产生，漏报与误报均保留显示。每个头的低决策余量标签单独显示，不从主预测统计中移除。</p><p class="muted">低决策余量 UNKNOWN（启发式）：|logit(p) − logit(clip(t))| &lt; 0.5，概率及阈值截断至 [10⁻⁷, 1−10⁻⁷]。这不是校准后的置信度，也不是安全拒答策略。12 个单元的类别为可见原生像素计数 0 / 1 / 2 / ≥3，不代表隐藏占据或物体数量。</p></section></main>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent), $=id=>document.getElementById(id);
const heads=['BODY','HEAD'];let filtered=[],index=0;
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function option(s,value,label){const o=el('option',label);o.value=value;s.append(o)}
function cond(c){let x=c.record.condition;return typeof x==='string'?x:JSON.stringify(x)}
function table(headers,rows){const t=el('table'),h=el('tr');headers.forEach(x=>h.append(el('th',x)));t.append(h);rows.forEach(row=>{const r=el('tr');row.forEach(x=>r.append(el('td',x)));t.append(r)});return t}
function cellName(i){return `${heads[Math.floor(i/6)]} ${i%6<3?'近段':'远段'} ${['左','中','右'][i%3]}`}
function thresholdLabel(t){return t>1?`${t.toPrecision(17)}（全阴性阈值）`:t.toFixed(4)}
option($('group'),'','全部场景组（原始顺序）');[...new Set(data.cases.map(c=>String(c.record.group_id)))].forEach(x=>option($('group'),x,x));
option($('condition'),'','全部条件');[...new Set(data.cases.map(cond))].forEach(x=>option($('condition'),x,x));
const totalRows=[];for(const arm of ['A','B'])data.summaries[arm].forEach((s,h)=>totalRows.push([arm,heads[h],s.tp,s.fn,s.fp,s.tn,s.unknown,s.truthUnknown]));
$('totals').append(table(['方法','头','命中 TP','漏报 FN','误报 FP','正确负例 TN','低余量 UNKNOWN','真值 UNKNOWN'],totalRows));
$('totals').append(el('p',`统计包含全部 ${data.cases.length} 帧，按冻结阈值；低余量列与主预测统计重叠，真值 UNKNOWN 不计入 TP/FN/FP/TN。`,'muted'));
$('sources').textContent=`缓存：${data.cache}；运行：${data.run}`;
function filter(){filtered=data.cases.filter(c=>(!$('group').value||String(c.record.group_id)===$('group').value)&&(!$('condition').value||cond(c)===$('condition').value));index=0;$('sample').replaceChildren();filtered.forEach((c,i)=>option($('sample'),String(i),`${c.record.sample_index} · ${c.record.name}`));render()}
function render(){const c=filtered[index];$('empty').hidden=!!c;$('main').hidden=!c;$('prev').disabled=!c||index===0;$('next').disabled=!c||index===filtered.length-1;$('position').textContent=`${c?index+1:0} / ${filtered.length}（全部 ${data.cases.length} 帧）`;if(!c)return;$('sample').value=String(index);$('rgb').src=c.rgb;$('caseTitle').textContent=c.record.name;$('caseInfo').textContent=`组 ${c.record.group_id} · ${cond(c)} · ${c.record.family||'未提供 family'} · sample_index ${c.record.sample_index}`;
$('truthNear').textContent=c.truth.near.map((v,h)=>`${heads[h]}：${v<0?'UNKNOWN':v?'有近场证据':'无近场证据'}`).join(' / ');
$('truthCounts').replaceChildren(table(['查询单元','可见像素计数真值'],c.truth.counts.map((v,i)=>[cellName(i),v<0?'UNKNOWN':v>=3?'≥3':v])));
for(const arm of ['A','B']){const target=$(arm),p=c.arms[arm];target.replaceChildren(el('h2',`${arm} · 预测支持图`));const img=el('img');img.className='frame';img.alt=`${arm} 的完整帧预测支持叠加`;img.src=p.overlay;target.append(img);
p.near.forEach((score,h)=>{const block=el('div',undefined,h===0?'body':'head');const positive=p.positive[h],y=c.truth.near[h],wrong=y>=0&&positive!==Boolean(y);block.append(el('p',`${heads[h]}：${positive?'近场证据阳性':'近场证据阴性'} · p=${score.toFixed(4)} · 阈值=${thresholdLabel(p.thresholds[h])}`));block.append(el('p',y<0?'真值 UNKNOWN，不能判定对错':wrong?(positive?'误报 FP':'漏报 FN'):(positive?'命中 TP':'正确负例 TN'),wrong?'bad':y<0?'muted':'good'));block.append(el('p',p.unknown[h]?'低决策余量 UNKNOWN（启发式）':'未触发低决策余量标签',p.unknown[h]?'unknown':'muted'));target.append(block)});
target.append(el('h2','12 个计数单元 · 类别概率'));target.append(table(['单元','P(0)','P(1)','P(2)','P(≥3)'],p.counts.map((row,i)=>[cellName(i),...row.map(v=>v.toFixed(3))])))} }
$('group').onchange=filter;$('condition').onchange=filter;$('sample').onchange=()=>{index=Number($('sample').value);render()};$('prev').onclick=()=>{index--;render()};$('next').onclick=()=>{index++;render()};filter();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be an absolute HTML path")
    payload = build_payload(args.cache, args.run)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__PAYLOAD__", encoded), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), eval_frames=len(payload["cases"]),
                          bytes=args.output.stat().st_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
