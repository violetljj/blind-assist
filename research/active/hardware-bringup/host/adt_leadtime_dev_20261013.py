"""Event-level warning lead time on ADT (revised after first run: lead counted from the last clear frame d > 2 m;
approaches already alerting while clear are reported separately as carried alerts) from per-frame alerts and GT corridor distance (plan PLAN_LEADTIME.md).
Usage: adt_leadtime_dev_20261013.py <adt_eval.json> <corridor_d.json> [alerts_key ...]
alerts_key: 'chain:0' (v3) / 'chain:1' (v6) / 'oracle:0' / 'oracle:1', or an external per-frame alert file via 'file:<path>:<name>'.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v2 as V2


def alerts_for(E, seq, key):
    arm, ro = key.split(':'); ro = int(ro)
    fr = E[seq]['frames']
    a = np.array([[f['arms'][arm][q][ro] for q in (0, 1)] for f in fr], bool)
    return V2.confirm(a[:, 0]) | V2.confirm(a[:, 1])


def evaluate(alerts_by_seq, Dd):
    ev, lead_d, lead_t, fa_eps, minutes, carry = [], [], [], 0, 0., []
    for seq, al in alerts_by_seq.items():
        t = np.array(Dd[seq]['t_ns']) / 1e9; d = np.array([np.inf if v is None else v for v in Dd[seq]['d']])
        minutes += (t[-1] - t[0]) / 60
        close = d <= 1.0
        i = 0
        while i < len(d):
            if close[i] and (i == 0 or not close[i - 1]):
                prev = np.flatnonzero((t >= t[i] - 5) & (t < t[i]) & (d > 2.0))
                if len(prev):
                    c = prev[-1]                                   # last clear frame (d > 2 m) before t0
                    if al[c]:
                        carry.append(1)                            # already alerting while clear: not a timely detection
                    else:
                        carry.append(0)
                        w = np.flatnonzero(al[c + 1:i + 1])
                        if len(w):
                            j = c + 1 + w[0]; ev.append(1); lead_d.append(d[j]); lead_t.append(t[i] - t[j])
                        else:
                            ev.append(0); lead_d.append(0.); lead_t.append(0.)
            i += 1
        clear = d > 2.0; k = 0
        while k < len(al):
            if al[k]:
                j = k
                while j < len(al) and al[j]: j += 1
                if clear[k:j].all(): fa_eps += 1
                k = j
            else:
                k += 1
    ev = np.array(ev); ld = np.array(lead_d); lt = np.array(lead_t)
    return dict(approaches=len(carry), carried_alert=int(sum(carry)), events=int(len(ev)), detected=int(ev.sum()), rate=float(ev.mean()) if len(ev) else None,
                lead_d_median=float(np.median(ld)) if len(ld) else None, lead_d_p25=float(np.percentile(ld, 25)) if len(ld) else None,
                lead_t_median=float(np.median(lt)) if len(lt) else None, lead_t_p25=float(np.percentile(lt, 25)) if len(lt) else None,
                fa_episodes=fa_eps, fa_per_min=fa_eps / minutes, minutes=minutes)


if __name__ == '__main__':
    E = json.loads(Path(sys.argv[1]).read_text()); Dd = json.loads(Path(sys.argv[2]).read_text())
    for key in sys.argv[3:] or ['chain:0', 'chain:1', 'oracle:0', 'oracle:1']:
        r = evaluate({s: alerts_for(E, s, key) for s in E}, Dd)
        print(key, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
