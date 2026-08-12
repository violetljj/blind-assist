# TARO O1R R10 source-only Phase A R1 recovery lock

R0 created its evidence root, then stopped before the first candidate because the
selected Python environment lacked `timm`. Its root contains only the execution
receipt, failure receipt, and manifest: candidate outputs = 0 and FARO reads = 0.
R0 remains immutable and cannot be resumed or rerun.

R1 uses a fresh exclusive root and performs the complete 710-frame Phase A from
the beginning. Before root creation it now fail-closes on the exact Python,
PyTorch, timm, CUDA, OpenCV, NumPy, and GPU identity. The official DepthART model
and checkpoint were successfully loaded in this environment without inference.

All original source/truth firewalls remain unchanged: candidates are sealed
before Apple-only features, FARO/truth reads are zero, `CLEAR` output and parent
scoring are forbidden, and no training is authorized. Failure does not restore
authority; overwrite, resume, repair in place, and rerun are forbidden.
