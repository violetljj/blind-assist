# Real positive approach role admission R0

Status: geometry-only data-role admission. This module must not import or run the
RCLE RGB algorithm.

The first executable step is the standalone `bootstrap_claim.py`. It exclusively
creates and fsyncs the hash-bound claim before any EVIMO2 path or network access.
All later source access is limited by
`RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0_CONTRACT_2026-07-27.json`.

Outputs belong under
`artifacts.local/evidence/rcle_phase_b_real_positive_approach_role_admission_r0/`.
The only valid scientific terminals are the two terminals in the machine
contract. No result from this module authorizes RGB algorithm implementation or
execution.
