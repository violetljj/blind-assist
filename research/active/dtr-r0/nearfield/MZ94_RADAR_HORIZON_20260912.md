# MZ94 experiment A: Radar horizon consistency

One fixed EXPLORE on consumed MZ90, before independent new-data comparison.
Only replace Radar admission. Keep ToF, 2-of-3 hysteresis and nonrecursive hold.
No coverage-authority change, calibration, learned head or source expansion here.

Radar tracks use frozen causal MZ91 mutual-unique range/angle/Doppler matching,
up to5 points over0.5s, max gap0.3s, reset on IMU gaps. For prediction require3
points spanning>=0.2s. Reuse point mean from MZ91 polar range/Doppler and angular
rate fits; convert its tangent to constant Cartesian velocity. No truth velocities.
Admit if current estimated r<3.6m and trajectory intersects the12deg route wedge
at depth>0.2m within[0,1]s. Current raw in-wedge occupancy at r<3.6 needs no history.
Missing transverse-motion evidence cannot seed a future-only crossing.

Important task semantics: GT includes t=0, so current in-wedge3.18–3.6m occupancy
also qualifies even if receding. This is route-wedge occupancy, not physical body
collision. Do not impose a universal closing-speed requirement contradicting GT.
Point estimation ignores posterior uncertainty; coarse bearing bias remains.

Controls: frozen matched_hold; range-only diagnostic changes only3.18 to3.6 and
keeps velocity<=-.35/bearing<=20. Range-only is an ablation, not the proposed
solution. A changes range, spatial corridor and motion eligibility together.
Any gain over baseline must also be compared with range-only before attributing
it to trajectory estimation. Feed admission directly to existing hysteresis;
radar_expert would incorrectly reapply the old gate.

Freeze cohorts from final audit:69 range-blocked FN (66 future-only),108 current
Radar FP and168 accompanying TP. Report recovery/removal/preservation on exact
frames, globally added/lost TP/FP and future-only TP. Do not redefine cohorts.
Primary proxy gate: F1 strictly higher, TP not lower, FP not higher; no additional
missed baseline event or>0.2s added delay, false segments/fragments not higher.
Report range-only comparison and ideal regression separately. Pass is Development
challenger only; partial benefit cannot promote full policy. No tuning or retry
for outcome; one fixed run per regime, both predictions sealed before evaluator.
CPU scalar scoring; no training or paid allocations. Experiment B is separate.

Artifacts: artifacts.local/work/mz94-radar-horizon-20260912/run-v1.
