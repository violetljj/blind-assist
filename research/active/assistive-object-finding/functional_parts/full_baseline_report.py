"""Summarize completed full-data runs; never loads test images or labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def model_summary(root, name):
    run = root / 'runs' / name
    terminal = load(run / 'terminal.json')
    config = load(run / 'config.json')
    path = run / 'progress.jsonl'
    curve = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []
    vals = [r for r in curve if r.get('stage') == 'validation']
    best = max(vals, key=lambda r: r['miou_foreground']) if vals else None
    last = vals[-1] if vals else None
    gain = None
    if last:
        old = [r for r in vals if r['epoch'] <= last['epoch'] - 10]
        if old:
            gain = max(r['miou_foreground'] for r in vals) - max(r['miou_foreground'] for r in old)
    independent = load(root / 'analysis' / name / 'summary.json')
    return {'terminal': terminal, 'config': config, 'validation_points': len(vals),
            'best_validation': best, 'last_validation': last,
            'best_miou_gain_in_last_10_epochs': gain,
            'convergence_note': 'Curve statistic only; a fixed epoch count is not proof of convergence',
            'independent_evaluation': independent}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    out = root / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    models = {name: model_summary(root, name) for name in ['segformer', 'dino']}
    summary = {'created_unix': time.time(), 'test_access': False,
               'scope': 'Full-image native semantic classes; not task role or instance-exclusive guidance',
               'models': models, 'paco': load(root / 'metadata/paco_summary.json'),
               'instructpart': load(root / 'metadata/instructpart_hf_census.json'),
               'iit_census': load(root / 'data/iit/native_census.json'),
               'iit_splits': load(root / 'data/iit/split_census.json'),
               'filename_split_audit': load(root / 'data/iit/filename_split_audit.json')}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# 全量功能区域基线：自动汇总', '',
             '本报告从完整运行记录生成；训练未完成时明确显示状态，不将启动或中期成绩称为充分训练完成。', '',
             '| 模型 | 训练状态 | train / val | 完成epoch | 最佳验证前景mIoU | 最后10epoch最佳值增量 |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    details = []
    for name, value in models.items():
        config = value['config'] or {}
        terminal = value['terminal'] or {}
        best = value['best_validation'] or {}
        last = value['last_validation'] or {}
        gain = value['best_miou_gain_in_last_10_epochs']
        score = f"{best['miou_foreground']*100:.3f}%" if best else 'NOT_RUN'
        inc = f'{gain*100:.3f}pp' if gain is not None else 'UNKNOWN'
        state = terminal.get('status', 'RUNNING' if config else 'NOT_RUN')
        lines.append(f"| {name} | {state} | {config.get('train_images','?')} / {config.get('val_images','?')} | {last.get('epoch',0)} | {score} | {inc} |")
        overall = (value['independent_evaluation'] or {}).get('overall')
        if overall:
            details += ['', f"{name}独立评分：{overall['images']}图；点命中{overall['point_hits']}/{overall['point_queries']}；原生标签误选{overall['native_label_mixups']}；同图多类查询联合成功{overall['joint_pair_successes']}/{overall['same_image_positive_class_pairs']}。"]
    lines += details
    lines += ['', '所有点指标按已知正类查询计数，不评价自然缺席；同图不同原生标签不等于真实任务双角色。',
              '官方划分的采集前缀可重叠；未做物理实例隔离的成绩不能称未见场景泛化。',
              'PACO语义部位配对与InstructPart当前发布完整性见metadata下的全量审计，不能把论文宣称规模当已取得数量。',
              '详细逐类、区域大小、多实例分层、混淆矩阵和分组置信区间位于各analysis目录。',
              '报告仅限公开原生区域的离线视觉能力，不代表真实视障操作成功。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(out), 'status': {k:(v['terminal'] or {}).get('status','INCOMPLETE') for k,v in models.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
