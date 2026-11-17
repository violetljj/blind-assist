"""Paired comparison (protocol v1.2: per band, plus the obstacle view) of two methods from saved rows_*.json.
Usage: paired_report.py <rows json> <method a (baseline)> <method b> [out json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import core


def main(path, a, b, out=None):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    c = core.compare(rows, a, b)
    print(f"{path}: {len(rows)} frames, common output {c['common_frames']}, no output {c['no_output']}")
    for band, d in list(c['detect'].items()) + [(k, v) for k, v in c.get('obstacle', {}).items()]:
        s = d['no_output_as_no_alert']
        print(f"  {band:8s} pos {d['positives']:4d}: tp {d['tp'][a]}/{d['tp'][b]} discordant {d['discordant_pos']['only_' + a]}/"
              f"{d['discordant_pos']['only_' + b]} p={d['discordant_pos']['p']:.3f} | neg {d['negatives']:4d}: fp {d['fp'][a]}/{d['fp'][b]} "
              f"discordant {d['discordant_neg']['only_' + a]}/{d['discordant_neg']['only_' + b]} p={d['discordant_neg']['p']:.2g}"
              f" | no-output-as-no-alert tp {s[a]['tp']}/{s[b]['tp']} of {s[a]['positives']}, fp {s[a]['fp']}/{s[b]['fp']} of {s[a]['negatives']}")
    for region, d in c['depth'].items():
        print(f"  {region:15s} blocks {d['blocks']:3d} AbsRel {d['absrel'][a]:.3f} -> {d['absrel'][b]:.3f} (b better in {d['blocks_b_better']})")
    if out:
        Path(out).write_text(json.dumps(c, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:5])
