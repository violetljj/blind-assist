"""Post-hoc obstacle view (protocol v1.2) on saved rows of earlier direction-A runs; descriptive only, no frozen verdict
changes. Usage: obstacle_relook.py <out json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import core

W = Path(__file__).resolve().parents[4] / 'artifacts.local/work'
CASES = [('A2 SANPO head', 'direction-a2-20261117/run/rows_sanpo_head_seg.json', [('tof_unidepth_out', 'hybrid_out'), ('tof_dav2s_out', 'hybrid_out')]),
         ('A2 Aria walks', 'direction-a2-20261117/run/rows_aria_gen2_walks.json', [('tof_unidepth_out', 'hybrid_out'), ('tof_dav2s_out', 'hybrid_out')]),
         ('Confirm SANPO chest', 'direction-a-confirm-20261117/run/rows_sanpo_chest_seg.json', [('tof_unidepth_out', 'hybrid_out'), ('tof_dav2s_out', 'hybrid_out')]),
         ('ADT indoor', 'direction-a-adt-20261118/run/rows_adt_apartment.json', [('tof_unidepth', 'hybrid_in'), ('tof_dav2s', 'hybrid_in')])]


def main(out_path):
    res = {}
    for name, path, pairs in CASES:
        rows = json.loads((W / path).read_text(encoding='utf-8'))
        for a, b in pairs:
            c = core.compare(rows, a, b); key = f'{name}: {b} vs {a}'
            res[key] = dict(common=c['common_frames'], obstacle=c['obstacle'],
                            bands={k: dict(positives=v['positives'], negatives=v['negatives'], tp=v['tp'], fp=v['fp']) for k, v in c['detect'].items()})
            o = c['obstacle']['ANY']; l = c['obstacle']['LOW_ONLY']
            print(f"{key:58s} common {c['common_frames']:4d} | ANY tp {o['tp'][a]}->{o['tp'][b]}/{o['positives']} "
                  f"(disc {o['discordant_pos']['only_' + a]}/{o['discordant_pos']['only_' + b]}) fp {o['fp'][a]}->{o['fp'][b]}/{o['negatives']} "
                  f"(disc {o['discordant_neg']['only_' + a]}/{o['discordant_neg']['only_' + b]}, p={o['discordant_neg']['p']:.2g}) | "
                  f"LOW_ONLY tp {l['tp'][a]}->{l['tp'][b]}/{l['positives']} | LOW tp {c['detect']['LOW']['tp'][a]}->{c['detect']['LOW']['tp'][b]}/{c['detect']['LOW']['positives']}")
    Path(out_path).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
