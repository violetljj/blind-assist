"""Summarize Track A scale v2 readout results into Markdown tables (no new computation)."""
import json
import sys
from pathlib import Path

CONDITIONS = [('primary-mount-10-snr6', 'M1 −10° / SNR6（主）'), ('secondary-mount-10-snr3', 'M1 / SNR3'),
              ('secondary-mount-10-snr12', 'M1 / SNR12'), ('secondary-mount0-snr6', 'M0 0° / SNR6')]
ARMS = ['B0', 'S1cell', 'S1', 'B1-R/noisy', 'S2/noisy', 'S2r4/noisy', 'S3/noisy', 'S2/GT', 'S3/GT']


def f(x, d=3):
    return '—' if x is None else f'{x:.{d}f}'


def main(root):
    root = Path(root)
    out = []
    results = {}
    for key, name in CONDITIONS:
        path = root/key/'result.json'
        if path.exists():
            results[key] = (name, json.loads(path.read_text(encoding='utf-8')))
    if 'primary-mount-10-snr6' in results:
        name, r = results['primary-mount-10-snr6']
        d = r['denominators']
        out.append(f"主条件：audit {d['audit_units']} 单位、{d['audit_main_frames']} 主子集帧；"
                   f"正/负查询 HEAD {d['audit_positive_queries']['HEAD']}/{d['audit_negative_queries']['HEAD']}，"
                   f"BODY {d['audit_positive_queries']['BODY']}/{d['audit_negative_queries']['BODY']}；calib {d['calib_units']} 单位。")
        out.append('\n| 主检验 | 组 | ΔAP | 95% 区间 | 为正单位 | p（单侧） | p（Holm） | 判定 |\n|---|---|---:|---|---|---:|---:|---|')
        for t in r['primary_tests']:
            out.append(f"| {t['comparison']} | {t['group']} | {t['delta']:+.3f} | [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | "
                       f"{t['units_positive']}/{t['units']} | {t['p_one_sided']:.4f} | {t['p_holm']:.4f} | "
                       f"{'成立' if t['holm_reject_at_0.05'] else '不成立'} |")
        out.append(f"\nv1.2 原规则（S2−B1-R 两组 95% 下界 > 0）：{'通过' if r['v12_frozen_criterion']['pass_gate'] else '未通过'}。"
                   f" τ 选择：" + '，'.join(f"{k} {v['tau']}" for k, v in r['selection'].items()))
        out.append('\n| 读出 | HEAD 宏 AP | BODY 宏 AP | HEAD 召回/FPR | BODY 召回/FPR | HEAD 事件召回 | BODY 事件召回 | HEAD 首报距离 m | BODY 首报距离 m |\n|---|---:|---:|---|---|---:|---:|---:|---:|')
        for arm in ARMS:
            if arm not in r['arms']:
                continue
            h, b = r['arms'][arm]['HEAD'], r['arms'][arm]['BODY']
            out.append(f"| {arm} | {f(h['macro_AP'])} | {f(b['macro_AP'])} | {f(h['recall'],2)}/{f(h['FPR'],3)} | "
                       f"{f(b['recall'],2)}/{f(b['FPR'],3)} | {f(h['events']['event_recall'],2)} | {f(b['events']['event_recall'],2)} | "
                       f"{f(h['events']['median_first_alert_distance_m'],2)} | {f(b['events']['median_first_alert_distance_m'],2)} |")
        out.append('\n| 读出 | HEAD 小块/现实/宽 AP | BODY 小块/现实/宽 AP |\n|---|---|---|')
        for arm in ('B0', 'S1cell', 'S1', 'S2/noisy', 'S3/noisy'):
            if arm not in r['arms']:
                continue
            s = [r['arms'][arm][g]['strata_AP'] for g in ('HEAD', 'BODY')]
            out.append(f"| {arm} | " + ' | '.join('/'.join(f(x[k]['AP'], 2) for k in ('tiny', 'realistic', 'wide')) for x in s) + ' |')
        if 'visibility_strata' in r:
            out.append('\n| 读出 | HEAD 不可见/可见 AP | BODY 不可见/可见 AP |\n|---|---|---|')
            for arm, g in r['visibility_strata'].items():
                out.append(f"| {arm} | {f(g['HEAD']['invisible']['AP'],2)}/{f(g['HEAD']['visible']['AP'],2)} "
                           f"({g['HEAD']['invisible']['positive']} 不可见正例) | {f(g['BODY']['invisible']['AP'],2)}/"
                           f"{f(g['BODY']['visible']['AP'],2)} ({g['BODY']['invisible']['positive']} 不可见正例) |")
    out.append('\n| 条件 | B0 | B1-R | S1 | S2 | S3 | S2−B1-R | S1−B0 | S3−S2 |\n|---|---|---|---|---|---|---|---|---|')
    for key, (name, r) in results.items():
        a = r['arms']
        cell = lambda arm: f"{f(a[arm]['HEAD']['macro_AP'])}/{f(a[arm]['BODY']['macro_AP'])}"
        deltas = {}
        for t in r.get('primary_tests', []):
            deltas.setdefault(t['comparison'], []).append(f"{t['delta']:+.3f}")
        out.append(f"| {name} | {cell('B0')} | {cell('B1-R/noisy')} | {cell('S1')} | {cell('S2/noisy')} | {cell('S3/noisy')} | "
                   + ' | '.join('/'.join(deltas.get(c, ['—'])) for c in ('S2/noisy - B1-R/noisy', 'S1 - B0', 'S3/noisy - S2/noisy')) + ' |')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1])
