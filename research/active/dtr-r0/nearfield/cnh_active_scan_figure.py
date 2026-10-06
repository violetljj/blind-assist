"""Thesis figure for the active-scan study; reads only result.json (no recomputation)."""
import json
from pathlib import Path

import cnh_tristate_thesis_figures as F

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT/'artifacts.local/work/cnh-active-scan-dev-20261006/result.json'
OUT = ROOT/'artifacts.local/work/cnh-active-scan-dev-20261006/figures'
ARMS = [('single/passive', '被动单路'), ('dual/passive', '被动双路'),
        ('single/prompt', '单路+提示\n(0.6 s反应)'), ('single/aligned', '单路头部正对\n(零延迟上限)')]
COLORS = ['#c0392b', '#2e86c1', '#d68910', '#1e8449']


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    F.style(plt, font_manager)
    res = json.loads(SRC.read_text(encoding='utf8'))
    G = res['groups']['all']; n = G['events']
    pts = [G['arms'][a][19] for a, _ in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    y = range(len(ARMS))[::-1]
    for yi, p, c in zip(y, pts, COLORS):
        u, s = p['unknown_miss'], p['silent']
        axes[0].barh(yi, u, color='#b3b6b7', height=.6); axes[0].barh(yi, s, left=u, color=c, height=.6)
        axes[0].text(u+s+.4, yi, f"无法判断 {u} / 静默 {s}　(及时 {p['timely']})", va='center', fontsize=10)
    axes[0].set_yticks(list(y), [l for _, l in ARMS]); axes[0].set_xlim(0, 34)
    axes[0].set_xlabel(f'未及时事件数（共 {n} 个接触事件）'); axes[0].set_title('(a) 截止时未及时事件的构成', loc='left')
    for yi, p, c in zip(y, pts, COLORS):
        full, late = 100*p['unknown_time']['value'], 100*p['unknown_time_late']['value']
        axes[1].barh(yi, full, color=c, height=.6, alpha=.85)
        axes[1].plot([late], [yi], marker='D', color='black', ms=6)
        axes[1].text(max(full, late)+3, yi, f'{full:.1f}%（后窗 {late:.1f}%）', va='center', fontsize=10)
    axes[1].set_yticks(list(y), ['']*len(ARMS)); axes[1].set_xlim(0, 100)
    axes[1].set_xlabel(f"畅通对照的无法判断时间（%，{G['controls']} 条×2.0 s）")
    axes[1].set_title('(b) 无法判断负担；◆为头部响应后窗口 f10–14', loc='left')
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    F.export(fig, OUT, 'fig4_active_scan_summary', plt)
    print(OUT/'fig4_active_scan_summary.png')


if __name__ == '__main__':
    main()
