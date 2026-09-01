# CARLA storage capacity extension for C40

Date: 2026-09-01

Decision: raise the CARLA experiment unique-byte cap from 100 GiB to 112 GiB.

## Decision-changing evidence

Before C40 capture, audited CARLA experiment storage contained
`100,499,871,816` unique bytes and the backing volume had
`223,906,783,232` free bytes. The normal 8 GiB reservation was refused by the
100 GiB unique-byte cap even though the volume free-space floor was not at
risk. The refusal occurred before an output directory or any fresh C40 pixel
was created.

The sealed-evidence PNG dedupe planner scanned `15,614` content candidates
covering `7,056,782,261` bytes and found zero duplicate groups and zero
reclaimable bytes. No payload was deleted, rewritten, or relabeled. The plan is
retained at
`artifacts.local/cleanup-records/carla-dedupe-before-c40-20260901-154500-plan`.

## Bounded change

- Maximum experiment unique bytes: 100 GiB -> 112 GiB
- Minimum backing-volume free bytes: unchanged at 100 GiB
- Default run reservation: unchanged at 8 GiB
- Overflow action: unchanged, `REFUSE_NEW_RUN`
- Automatic payload deletion: unchanged, disabled

With the new cap, the same 8 GiB reservation projects
`109,089,806,408` unique bytes and `215,093,882,880` free volume bytes at the
post-change guard. Both guards remain fail-closed. This extension preserves every prior
cohort and admits the already-preregistered C40 fresh-source capture; it is not
a general waiver of storage accounting.
