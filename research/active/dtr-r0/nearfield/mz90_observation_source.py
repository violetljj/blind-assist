"""MZ90 paired analytic observation source; not a physical sensor simulator.

Eight horizontal 5.625-degree bins are a 45-degree surrogate, NOT VL53L8CX 8x8.
Objects are point supports in a common 2D world. No body/head inference is made.
delta_yaw is an incremental angle in degrees; unavailable increments are zero.
Source metadata and evaluator truth must never be given to a predictor.
"""
from __future__ import annotations

import numpy as np

DT = 0.1
FRAMES = 40
EPISODES = 48
WEARER_SPEED = 0.7
TOF_FOV = 45.0
TOF_BIN = TOF_FOV / 8
REGIMES = ("ideal", "sensor_proxy")


def corridor_intersection(x, z, vx, vz, horizon=1.0):
    """Exact linear-segment clipping against z > .2, |x| <= tan(12deg) z.

    Checking the midpoint of the clipped interval enforces strict forward depth
    and excludes a segment that only touches z=.2 without entering the corridor.
    """
    slope = np.tan(np.deg2rad(12.0))
    lo, hi = 0.0, float(horizon)
    for a, b in ((z - 0.2, vz), (slope*z - x, slope*vz - vx),
                 (slope*z + x, slope*vz + vx)):
        if abs(b) < 1e-12:
            if a < 0:
                return False
        elif b > 0:
            lo = max(lo, -a/b)
        else:
            hi = min(hi, -a/b)
        if lo > hi:
            return False
    return z + vz * ((lo + hi) / 2.0) > 0.2


def geometric_hazard(x, z, vx, vz):
    r = np.hypot(x, z)
    angle = np.degrees(np.arctan2(x, z))
    current = z > 0.2 and r < 3.18 and abs(angle) <= 12.0
    future = z > 0.2 and r < 3.6 and corridor_intersection(x, z, vx, vz)
    return bool(current or future)


def build_source(seed=90012):
    """Create 48 reproducible world episodes, before any sensor realization."""
    rng = np.random.default_rng(seed)
    scenes = []
    for i in range(EPISODES):
        objects = []
        for _ in range(0 if i % 8 == 0 else int(rng.integers(1, 4))):
            moving = bool(rng.random() < 0.55)
            objects.append(dict(x=float(rng.uniform(-3, 3)),
                                z=float(rng.uniform(2.5, 5.5)),
                                vx=float(rng.uniform(-1.5, 1.5)) if moving else 0.0,
                                vz=float(rng.uniform(-0.2, 0.5)) if moving else 0.0,
                                reflectivity=float(rng.uniform(0.3, 1.0))))
        sign = int(rng.choice([-1, 1]))
        scenes.append(dict(
            episode_id=f"e{int(rng.integers(0, 2**63)):016x}", scene=i,
            variant=i % 6, sensor_seed=int(rng.integers(0, 2**63)),
            objects=objects, yaw_amplitude=float(rng.uniform(15, 25)),
            yaw_omega=float(rng.uniform(1.3, 2.5)), extrinsic_deg=2.0*sign,
            imu_bias_dps=2.0*sign,
            radar_ambiguity_deg=float(rng.choice([0, -10, 10])),
            persistent_clutter=(i % 4 == 0),
            clutter_x=float(rng.uniform(-2.5, 2.5)),
            clutter_z=float(rng.uniform(2.0, 5.0)),
            imu_gap_start=18 if i % 2 == 0 else -1))
    return scenes


def _yaw(scene, t):
    return scene["yaw_amplitude"] * np.sin(scene["yaw_omega"] * t)


def materialize(source, regime):
    """Return (raw observation arrays, evaluator-only arrays), no disk writes.

    Proxy Radar packets are always received (no dropout rate was stipulated),
    independently of per-object missed detections. ToF packet loss is separate
    from per-beam missing returns. Persistent Radar clutter is a stationary
    world-space phantom with ordinary relative radial velocity; it is NOT truth.
    Transient clutter is generated uniformly in raw sensor range/azimuth.
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime: {regime}")
    proxy = regime == "sensor_proxy"
    n = len(source) * FRAMES
    raw = dict(episode_id=np.empty(n, dtype="U32"), time_s=np.empty(n),
               tof_packet_received=np.ones(n, dtype=bool),
               tof_range_m=np.full((n, 8), np.nan),
               tof_theta_deg=np.full((n, 8), np.nan),
               tof_range_sigma_m=np.full((n, 8), np.nan),
               tof_status=np.full((n, 8), 255, dtype=np.uint8),
               radar_packet_received=np.ones(n, dtype=bool),
               radar_range_m=np.full((n, 4), np.nan),
               radar_velocity=np.full((n, 4), np.nan),
               radar_angle=np.full((n, 4), np.nan),
               radar_valid=np.zeros((n, 4), dtype=bool),
               delta_yaw=np.zeros(n), delta_pitch=np.zeros(n),
               imu_valid=np.ones(n, dtype=bool))
    evaluator = dict(truth=np.zeros(n, dtype=bool),
                     any_direct_truth=np.zeros(n, dtype=bool),
                     future_only_truth=np.zeros(n, dtype=bool),
                     scene=np.empty(n, dtype=int), variant=np.empty(n, dtype=int),
                     truth_height=np.full(n, "UNKNOWN", dtype="U7"))
    for e, scene in enumerate(source):
        rng = np.random.default_rng(scene["sensor_seed"])
        for f in range(FRAMES):
            j, t = e*FRAMES + f, f*DT
            raw["episode_id"][j] = scene["episode_id"]
            raw["time_s"][j] = t
            evaluator["scene"][j], evaluator["variant"][j] = scene["scene"], scene["variant"]
            yaw = _yaw(scene, t)
            sensor_yaw = yaw + (scene["extrinsic_deg"] if proxy else 0.0)
            if f:
                delay = 0.02 if proxy else 0.0
                # No negative-time pose observations; yaw starts at zero.
                raw["delta_yaw"][j] = _yaw(scene, max(0, t-delay)) - _yaw(scene, max(0, t-DT-delay))
                if proxy:
                    raw["delta_yaw"][j] += DT*(scene["imu_bias_dps"] + rng.normal(0, 2))
            if proxy and scene["imu_gap_start"] <= f < scene["imu_gap_start"]+2 and scene["imu_gap_start"] >= 0:
                raw["imu_valid"][j] = False
                raw["delta_yaw"][j] = 0.0
            returns = []
            for obj in scene["objects"]:
                x, z = obj["x"] + obj["vx"]*t, obj["z"] + (obj["vz"]-WEARER_SPEED)*t
                vx, vz = obj["vx"], obj["vz"]-WEARER_SPEED
                r = float(np.hypot(x, z))
                a = float(np.degrees(np.arctan2(x, z)) - sensor_yaw)
                rv = (x*vx + z*vz)/max(r, 1e-12)
                evaluator["truth"][j] |= geometric_hazard(x, z, vx, vz)
                evaluator["any_direct_truth"][j] |= (z > 0.2 and r < 3.18 and abs(np.degrees(np.arctan2(x, z))) <= 12.0)
                returns.append((r, a, rv, obj["reflectivity"]))
            evaluator["future_only_truth"][j] = evaluator["truth"][j] and not evaluator["any_direct_truth"][j]
            raw["tof_packet_received"][j] = not proxy or rng.random() >= 0.1
            if not raw["tof_packet_received"][j]:
                raw["tof_status"][j] = 0
            else:
                # Closest among detected supports, not necessarily closest truth.
                for r, a, _, reflectivity in sorted(returns):
                    if not (-TOF_FOV/2 <= a < TOF_FOV/2):
                        continue
                    b = int((a+TOF_FOV/2)//TOF_BIN)
                    pd = (0.9 if r <= 2 else 0.65 if r <= 3 else 0.35 if r <= 4 else 0.0)*reflectivity
                    if proxy and rng.random() >= pd:
                        continue
                    if raw["tof_status"][j, b] == 5:
                        continue
                    raw["tof_range_m"][j, b] = max(0.03, np.round((r+rng.normal(0, 0.05))/0.03)*0.03) if proxy else r
                    raw["tof_theta_deg"][j, b] = -TOF_FOV/2+(b+0.5)*TOF_BIN if proxy else a
                    raw["tof_range_sigma_m"][j, b] = 0.05 if proxy else 0.0
                    raw["tof_status"][j, b] = 5
            radar = []
            for r, a, rv, _ in returns:
                if abs(a) <= 60 and (not proxy or rng.random() < 0.8):
                    radar.append((r, a, rv))
            if proxy and scene["persistent_clutter"]:
                x, z = scene["clutter_x"], scene["clutter_z"]-WEARER_SPEED*t
                r = float(np.hypot(x, z))
                a = float(np.degrees(np.arctan2(x, z))-sensor_yaw)
                if abs(a) <= 60:
                    radar.append((r, a, -WEARER_SPEED*z/max(r, 1e-12)))
            measured = []
            for r, a, rv in radar:
                if proxy:
                    r = max(0.05, np.round((r+rng.normal(0, 0.06))/0.05)*0.05)
                    a = np.round((a+rng.normal(0, 2))/10)*10 + scene["radar_ambiguity_deg"]
                    rv = np.round((rv+rng.normal(0, 0.08))/0.1)*0.1
                measured.append((r, a, rv))
            if proxy:
                for _ in range(rng.poisson(0.15)):
                    measured.append((float(np.round(rng.uniform(0.4, 5)/0.05)*0.05),
                                     float(np.round(rng.uniform(-60, 60)/10)*10),
                                     float(np.round(rng.uniform(-1, 1)/0.1)*0.1)))
            for k, (r, a, rv) in enumerate(sorted(measured)[:4]):
                raw["radar_range_m"][j, k] = r
                raw["radar_angle"][j, k] = a
                raw["radar_velocity"][j, k] = rv
                raw["radar_valid"][j, k] = True
    return raw, evaluator
