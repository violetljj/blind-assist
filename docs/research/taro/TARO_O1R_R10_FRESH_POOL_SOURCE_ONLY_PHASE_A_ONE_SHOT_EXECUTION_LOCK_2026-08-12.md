# TARO O1R R10 fresh-pool source-only Phase A lock

The exact 32-parent inventory contains 710 pose-bounded frames and 6,390 fixed
queries. This one-shot lock first runs the registered official DepthART candidate
from RGB and bound intrinsics, seals all 710 candidates, then reads only Apple
depth/confidence plus geometry to build source-side query features.

FARO and truth reads remain exactly zero, no parent scoring occurs in this phase,
`CLEAR` output is forbidden, and no threshold fitting or training is authorized.
The expected normal completion has `5F+4 = 3,554` files before its manifest.

The output root is consumed on exclusive creation. Failure does not restore
authority; overwrite, resume, repair in place, and rerun are forbidden. A pass
permits only a separately locked source-only 32-to-8 selection using the frozen
R9 selector.
