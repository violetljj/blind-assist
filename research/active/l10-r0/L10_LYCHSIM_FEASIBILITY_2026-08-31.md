# L10 LychSim feasibility screen (2026-08-31)

Status: `CONDITIONAL_SECONDARY_ONLY`

## Decision

Do not replace SEVN/PanoLab with LychSim. LychSim may enter one bounded smoke
test as a secondary synthetic counterfactual lab; it receives no L10 mainline
authority unless that test passes unchanged.

This is a documentation and source-code screen, not an installed-runtime result.

## Evidence

- The [public repository](https://github.com/wufeim/LychSim) was created on
  2026-05-12, has one `v1.0.0` release, a small commit history, and asks for
  maintainers. The release has no Windows binary asset.
- The [Windows installation](https://wufeim.github.io/LychSim/tutorials/installation.html)
  requires Unreal Engine 5.5.4, Visual Studio project generation, and local
  plugin compilation. The [no-UE quick start](https://wufeim.github.io/LychSim/tutorials/quick_start.html)
  currently supplies a Linux demo package only.
- The [Python API](https://wufeim.github.io/LychSim/docs/python_api.html)
  exposes camera pose, RGB, centimetre depth, instance/element segmentation,
  point maps, pause, and resume. `PAN/SWEEP` therefore map directly. Camera
  relocation can approximate `APPROACH/SIDESTEP`, but it is teleportation rather
  than continuous pedestrian motion or collision-checked traversal. No public
  fixed-step, seed, or exact reset contract was found.
- Public scene content is still narrow: the documented outdoor path depends on
  Unreal/Fab assets, while fuller scene/object/layout support remains on the
  project roadmap.
- The local RTX 5060 Laptop GPU has 8 GB VRAM, at the Unreal recommended VRAM
  line, but 16 GB system RAM is half of Epic's 32 GB recommendation. A small
  packaged scene may run; editor + compiler + street scene + model is a material
  memory risk. See [Epic hardware guidance](https://dev.epicgames.com/documentation/unreal-engine/hardware-and-software-specifications-for-unreal-engine).

## Bounded smoke test

Use one ten-metre scene with one target door, one decoy door, one sign, one
static pedestrian occluder, and one camera. Execute
`HOLD -> PAN/SWEEP -> SIDESTEP -> APPROACH`; save pose, RGB, depth, and instance
mask after each step, reset, and replay once.

Conditional `GO` requires all of:

1. UE 5.5.4 and the plugin compile on Windows without source patches.
2. Twenty 640x480 action-to-RGB/depth/mask cycles finish without crash or OOM,
   with p95 latency at most two seconds.
3. Pose readback error is at most 1 cm / 0.1 degrees; a 5 m plane has at most
   5 cm depth error; the target-door instance ID remains stable.
4. Two warm replays have identical depth/masks and RGB SSIM at least 0.995.
5. `HOLD` does not improve target visibility, while at least one frozen
   `PAN/SIDESTEP` action improves it by at least 25 percentage points.

Any failed core condition, required paid asset, RAM upgrade, or continuing
plugin repair is `NO_GO`. A pass authorizes only synthetic mechanism and
counterfactual work. It cannot establish real entrance identity, public access,
arrival, handoff, user benefit, or safety.
