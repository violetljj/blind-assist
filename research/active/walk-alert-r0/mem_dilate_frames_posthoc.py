"""Post-hoc (descriptive): frame-level false-alert rate vs FA episode count at tau = 5, cache_mem_dilate arms, blind walkers.
Checks whether partial de-dynamic splits long trail alerts into more FA episodes."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, walk_alert_policy_sanpo as w
w.V2 = True; w.CACHE_DIR = 'cache_mem_dilate'; w.BW_SPEEDS.update(w.bw_speeds())
data = w.load(sys.argv[1])
for a in range(8):
    fa_frames = alert_frames = km = eps_fa = 0
    for s, r in data.items():
        for e in w.episodes(s, r, 'blind'):
            al = w.policy_alerts('B1', 5.0, e, a) & e['valid']; rr = e['rows']
            clean = ~(rr[:, 0] <= w.H) & ~(rr[:, 2] > .5) & ~w.weak_mask(e)
            fa_frames += int((al & clean).sum()); alert_frames += int(al.sum())
            sc = w.score(e, al); eps_fa += sc[3]; km += sc[4]
    print(a, 'false-alert frames/km %.1f' % (fa_frames / km), 'FA episodes/km %.1f' % (eps_fa / km),
          'frames per episode %.2f' % (fa_frames / max(eps_fa, 1)), flush=True)
