# MZ69 execution

One registered frozen pass and one independent saved-output score completed. No training, new cutoff, old-source replay or permanent dense cache.

Actual model run: 76.231050 s; wrapper: 79.113904 s. Score: 3.465807 s. Backend: CUDA frozen inference; CPU compact-source I/O; device: NVIDIA GeForce RTX 5060 Laptop GPU.

Timing components (s): `{"rgb_decode": 23.263771500089206, "visual_views": 25.257364099816186, "fixed_readouts": 20.20810890015855}`. RGB loads: 4096; encoded RGB bytes read: 1698086934; per-view frames: `{"global256x144_BOX": 4096, "crop224x224": 4096, "full640x360": 4096}`. Shared views and avoided cache allocation are actual execution behavior, not a separately measured speedup.

Independent checks: `{"status": "PASS", "frames": 4096, "scalar_known_decisions": 1048576, "scalar_native_winner_lookups": 262144, "scalar_candidate_and_union_values": 458752, "scalar_exchange_counts": true, "source_pairs": 2048, "original_cutoffs_exact": true, "packet_scalar_parity": true, "native_winner_argmax_recomputed": false, "MZ37_retained_by_all_methods": true, "all_groups_retained": true, "unknown_separate": true, "fits": 0, "inference_frames": 0, "cutoff_searches": 0, "RGB_reads": 0, "raw_native_reads": 0}`.

| Artifact under artifacts.local/work/mz69-topology-transfer-20260911 | SHA256 |
| --- | --- |
| run-v1/receipt.json | `b65fa2e2c45c2014761fb03254267b38307c1ceaf49c846653da562e6ffdf259` |
| score-v1/receipt.json | `37565ab3a6d254496f868768b4e19f3b50aee1099a37f3ddfce4f5df40905eeb` |
| score-v1/result.json | `21fe45f9d6dabe4aebe1cb8b1ee7e066e39c0be73eaf7edd1c3c4a9cbb544a7b` |
| score-v1/audit.json | `9428a5955b48b7bfe515628e4dddf462f81cefee40f0c48f35e62101bbafcaf3` |
| ranking-diagnostic-v1/receipt.json | `577cb32dc650928756a1044705e14ace8e108050e43a88deafec93f0ba2c6c56` |
| ranking-diagnostic-v1/result.json | `02185fe39f9f6b4eaf91ecce54cf174fdf86a3fc8365ebff070883199c857ac7` |
| execution-v1.json | `bdb7410ecdf2b68b3e18b9b56fa903654b87fb98f55cd0a597cd876347f3a030` |
| score-execution-v1.json | `90b1f8860ac9276815a905569f1ddc63bfee20ba4e95118e0a5fdefa73e988c6` |
| handle-release.json | `c40177ffdab978bb7c04981758e508962f937e76acdd157c9f3fe41ebcb36705` |
| inputs.json | `ca80fb3cda67a70ae527b00323d8851589434e0720cf62cb09becd6817bce415` |
| registration.json | `dd6419e0388b906fc09bf2f4937cc0c49b1267d43ad0bd0828ba1d7d5303a055` |

Full result: `score-v1/result.json`; exact exchanges and support pairs: `score-v1/paired-events.json`. The source-index remains `23aa6f552e38a0534052d2723b9a447ee91574bacacd503ecc64957c81b6560b`. Original checkpoint/cutoff bytes and all input bindings are retained in the run receipt.

Task-owned model and scorer processes exited with code 0 and no survivors; image/archive handles closed. No permanent dense cache was created. Preparation CPU fixture failures, if any, remain in preparation evidence and do not represent a model run. No independent original-input neural re-encode or dense-logit argmax replay was performed.
