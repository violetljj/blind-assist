# CARLA storage capacity extension

Date: 2026-09-01

Decision: raise the CARLA experiment unique-byte cap from 80 GiB to 100 GiB.

## Decision-changing evidence

Before C34 capture, audited CARLA experiment storage contained
`80,766,761,041` unique bytes and the backing volume had
`251,778,580,480` free bytes. A 6 GiB reservation was refused by the 80 GiB
unique-byte cap even though the volume free-space floor was not at risk.

The sealed-evidence PNG dedupe planner scanned the eligible corpus and found
zero duplicate identities and zero reclaimable bytes. No payload was deleted or
rewritten.

## Bounded change

- Maximum experiment unique bytes: 80 GiB -> 100 GiB
- Minimum backing-volume free bytes: unchanged at 100 GiB
- Default run reservation: unchanged at 8 GiB
- Overflow action: unchanged, `REFUSE_NEW_RUN`
- Automatic payload deletion: unchanged, disabled

With the new cap, the post-change guard plus an 8 GiB reservation projects
`89,356,717,196` unique bytes and `243,188,621,312` free volume bytes. Both
guards remain fail-closed. This extension preserves prior evidence and admits
one bounded fresh-source capture; it is not a general waiver of storage
accounting.
