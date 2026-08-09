# Assistive Geometry R2 F0 synthetic factor geometry canary protocol

## Decision

Only F0 is executable. F1 and every later model, dataset, teacher, temporal, mobile, Calibration and Confirmation action remain unauthorized.

F0 asks one question: if continuous geometric factor evidence is correct—or is corrupted in a precisely declared way—does the frozen deterministic reducer produce the analytically correct clearance interval and tri-state geometry?

This is not a model stage. It uses no RGB, learned parameter, real dataset, prior B1 outcome, training step or task-gate tuning.

## Frozen construction

`GeometryR2Reducer` consumes only bound input geometry, depth/scale intervals, support-surface uncertainty, and obstacle/boundary evidence. It is the sole producer of final task geometry.

- `CLEAR_OBSERVED`: no admissible obstacle interval can occupy the body band inside the horizon.
- `OCCUPIED_OBSERVED`: positive evidence, guaranteed lateral overlap, and the upper forward-distance bound all prove occupancy.
- `UNKNOWN`: global geometry is invalid, a local factor is missing, boundary coverage is incomplete, or the interval can occupy the band but cannot prove it.

The anti-A0 invariant is frozen as:

> Without sufficient positive occupancy evidence, uncertainty may not produce `OCCUPIED_OBSERVED`.

When uncertainty increases, a definite state may remain unchanged or degrade to `UNKNOWN`; uncertainty alone may never turn `CLEAR_OBSERVED` into `OCCUPIED_OBSERVED`.

## Frozen fixtures

The 23-case suite covers:

- perfect, biased, noisy and missing depth/scale;
- flat, sloped, missing, uncertain and incorrectly oriented support;
- sharp, weak, discontinuous, texture-only, partial and blurred boundary evidence;
- exact reducer closure, horizon nesting, band ownership, orientation parity and deterministic replay;
- nine anti-collapse cases including open flat space, an open center corridor between side obstacles, far obstacles, missing local depth, degraded support, weak/partial/blurred boundaries and insufficient scale evidence;
- a negative control that injects a learned final-task shortcut and must be rejected.

The fixture bytes, reducer, runner, tests, and parent R2 hypothesis are SHA-256-bound in the JSON protocol before outcome access.

## Kill gate

All ten gates are conjunctive. Any missing, undefined, mismatched, non-deterministic or anti-monotonic result kills `geometry_r2_interval_reducer_f0_v1`. A failed run may not be rescued by changing thresholds, adding training, adding a task head or consuming real data.

A PASS establishes only frozen synthetic reducer mechanics. It may authorize writing a separate F1 TRAIN-only protocol, but F1 execution authority remains `false`.

The machine-readable authority, hashes, exact constants, gates and terminals are normative in the companion JSON.
