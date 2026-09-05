"""Evaluate actual online street trajectories and render paired sensor playback.

Only the evaluator reads scenario actors and expected controls. All results are
scripted synthetic Development using declared geometry proxies, not human safety.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import io
import json
import math
from pathlib import Path
import statistics

from street_scenarios import actors_at, contacts_for_step, scenario_catalog


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _command(frame):
    return frame.get("response", {}).get("command", {})


def _sources(frame):
    command = _command(frame)
    prediction = frame.get("response", {}).get("prediction", {})
    return [name for name, positive in (
        ("DTR_X73", command.get("dtr_route_risk", prediction.get("route_risk", False))),
        ("OBSERVED_DEPTH", command.get("depth_near_risk", False))) if positive]


def _pose_at(frames, t):
    if t <= frames[0]["time_s"]:
        return frames[0]["ego"]
    for a, b in zip(frames, frames[1:]):
        if t <= b["time_s"]:
            u = (t - a["time_s"]) / (b["time_s"] - a["time_s"])
            return {k: a["ego"][k] + u * (b["ego"][k] - a["ego"][k]) for k in ("x_m", "y_m")}
    return frames[-1]["ego"]


def evaluate_episode(spec, episode):
    if episode is None:
        return {"status": "INCOMPLETE", "reason": "Missing episode", "success": False}
    frames = episode.get("frames", [])
    if len(frames) < 2:
        return {"status": "INCOMPLETE", "reason": "Fewer than two frames", "success": False}
    if any(b["time_s"] <= a["time_s"] for a, b in zip(frames, frames[1:])):
        return {"status": "INVALID", "reason": "Non-increasing timestamps", "success": False}
    contacts, mismatches, motion_mismatches = [], [], []
    latencies = []
    for i, frame in enumerate(frames):
        latency = frame.get("response", {}).get("elapsed_s")
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            latencies.append(latency)
        if not i:
            continue
        previous = frames[i - 1]
        hits = contacts_for_step(spec, previous["time_s"], frame["time_s"], previous["ego"], frame["ego"])
        contacts.extend(hits)
        reported = frame.get("contacts_since_previous")
        if reported is None:
            mismatches.append({"sample_index": frame["sample_index"], "reason": "Missing contact receipt"})
        else:
            keyed = lambda rows: sorted((h["actor_id"], h["contact_type"], round(h["time_s"], 5)) for h in rows)
            if keyed(hits) != keyed(reported):
                mismatches.append({"sample_index": frame["sample_index"], "reason": "Swept contact receipt mismatch"})
        applied = frame.get("applied_command", {})
        dt = frame["time_s"] - previous["time_s"]
        if "vx_mps" in applied and "vy_mps" in applied:
            error = math.hypot(frame["ego"]["x_m"] - previous["ego"]["x_m"] - dt * applied["vx_mps"],
                               frame["ego"]["y_m"] - previous["ego"]["y_m"] - dt * applied["vy_mps"])
            # Goal clamping may shorten the final integration step.
            if error > 0.015 and not (i == len(frames) - 1 and episode.get("goal_reached")):
                motion_mismatches.append({"sample_index": frame["sample_index"], "position_error_m": error})
    forward = frames[-1]["ego"]["x_m"] - frames[0]["ego"]["x_m"]
    target = episode.get("goal_forward_m", 8.0)
    reached = bool(episode.get("goal_reached")) and forward >= target - 0.05
    declared_goal_consistent = bool(episode.get("goal_reached")) == (forward >= target - 0.05)
    complete = bool(episode.get("completed"))
    valid = not mismatches and not motion_mismatches and declared_goal_consistent
    status = "COMPLETE" if complete and valid else "INVALID" if not valid else "INCOMPLETE"
    events = [{"time_s": f["time_s"], "sources": _sources(f), "action": _command(f).get("action")}
              for f in frames if _sources(f)]
    path = sum(math.hypot(b["ego"]["x_m"] - a["ego"]["x_m"],
                          b["ego"]["y_m"] - a["ego"]["y_m"])
               for a, b in zip(frames, frames[1:]))
    return {"episode_id": episode["episode_id"], "status": status, "frames": len(frames),
            "success": status == "COMPLETE" and reached and not contacts,
            "goal_reached": reached, "declared_goal_consistent": declared_goal_consistent,
            "forward_progress_m": round(forward, 5), "goal_forward_m": target,
            "path_length_m": round(path, 5), "duration_s": frames[-1]["time_s"] - frames[0]["time_s"],
            "contact": bool(contacts), "first_contact_s": min((c["time_s"] for c in contacts), default=None),
            "contact_actors": sorted({c["actor_id"] for c in contacts}),
            "contact_types": sorted({c["contact_type"] for c in contacts}),
            "contact_intervals": len(contacts), "contacts": contacts,
            "tactile_ground_contact": any(c["actor_id"] == "tactile_ground" for c in contacts),
            "contact_receipt_mismatches": mismatches, "applied_motion_mismatches": motion_mismatches,
            "alerts": events, "alert_sources": sorted({s for e in events for s in e["sources"]}),
            "first_alert_s": events[0]["time_s"] if events else None,
            "measured_worker_latency_s": {"samples": len(latencies),
                "median": statistics.median(latencies) if latencies else None,
                "p95": sorted(latencies)[max(0, math.ceil(len(latencies) * .95) - 1)] if latencies else None,
                "mean": statistics.mean(latencies) if latencies else None},
            "measured_worker_throughput_fps": len(latencies) / sum(latencies) if sum(latencies) > 0 else None}


def causal_trajectory_check(spec, baseline, assisted):
    """Measured chronology check, not a randomized causal identification claim."""
    if not baseline or not assisted or not baseline.get("frames") or not assisted.get("frames"):
        return {"status": "INCOMPLETE", "trajectory_changed_after_sensor_alert": False}
    left, right = baseline["frames"], assisted["frames"]
    deviations = []
    chain_mismatches = []
    for i, frame in enumerate(right):
        # Compare trajectories only over the observed shared time range.
        if frame["time_s"] <= left[-1]["time_s"] + 1e-8:
            nominal = _pose_at(left, frame["time_s"])
            if math.hypot(frame["ego"]["x_m"] - nominal["x_m"], frame["ego"]["y_m"] - nominal["y_m"]) > 0.025:
                deviations.append(frame["time_s"])
        if i:
            previous = _command(right[i - 1])
            applied = frame.get("applied_command", {})
            if any(abs(applied.get(k, float("inf")) - previous.get(k, float("-inf"))) > 1e-6
                   for k in ("vx_mps", "vy_mps")):
                chain_mismatches.append(frame["sample_index"])
    first = deviations[0] if deviations else None
    triggers = [f for f in right if first is not None and f["time_s"] < first - 1e-8 and _sources(f)
                and (abs(_command(f).get("vx_mps", spec["ego_speed_mps"]) - spec["ego_speed_mps"]) > .01
                     or abs(_command(f).get("vy_mps", 0)) > .01)]
    return {"status": "COMPLETE", "trajectory_changed": bool(deviations),
            "first_actual_deviation_s": first,
            "trigger_time_s": triggers[0]["time_s"] if triggers else None,
            "trigger_sources": _sources(triggers[0]) if triggers else [],
            "response_to_applied_command_mismatches": chain_mismatches,
            "trajectory_changed_after_sensor_alert": bool(triggers) and not chain_mismatches,
            "claim": "Observed sensor-alert/command/motion chronology; no counterfactual human-safety claim"}


def evaluate(root):
    specs = _read(root / "evaluator/scenarios.json")
    if isinstance(specs, dict):
        specs = specs["scenarios"]
    receipt = _read(root / "run.json") if (root / "run.json").exists() else {"status": "INCOMPLETE"}
    episodes = [_read(p) for p in sorted((root / "evaluator/episodes").glob("*.json"))]
    indexed, duplicates = {}, []
    for ep in episodes:
        key = (ep["scenario_id"], ep["arm"])
        if key in indexed:
            duplicates.append(list(key))
        indexed[key] = ep
    pairs = []
    for spec in specs:
        original = indexed.get((spec["id"], "OPEN_LOOP"))
        assisted = indexed.get((spec["id"], "ASSISTED"))
        a, b = evaluate_episode(spec, original), evaluate_episode(spec, assisted)
        evaluable = a["status"] == "COMPLETE"
        contrast = (a.get("contact") == spec["expected_open_loop_contact"] and
                    (not spec["expected_open_loop_contact"] or
                     spec["expected_contact_type"] in a.get("contact_types", []))) if evaluable else None
        pairs.append({"scenario_id": spec["id"], "family": spec["family"], "control": spec["control"],
                      "expected_open_loop_contact": spec["expected_open_loop_contact"],
                      "open_loop_contrast_pass": contrast, "OPEN_LOOP": a, "ASSISTED": b,
                      "assisted_delay_s": b.get("duration_s", 0) - a.get("duration_s", 0)
                         if a.get("goal_reached") and b.get("goal_reached") else None,
                      "causal_trajectory": causal_trajectory_check(spec, original, assisted)})
    required_ids = {s["id"] for s in scenario_catalog()}
    supplied_ids = [s["id"] for s in specs]
    catalog_complete = (len(supplied_ids) == len(required_ids) and set(supplied_ids) == required_ids and
                        sum(s["expected_open_loop_contact"] is True for s in specs) == 4 and
                        sum(s["expected_open_loop_contact"] is False for s in specs) == 4)
    run_complete = receipt.get("status") in ("PASS", "COMPLETE", "COMPLETED")
    all_complete = all(p[arm]["status"] == "COMPLETE" for p in pairs for arm in ("OPEN_LOOP", "ASSISTED"))
    complete = catalog_complete and run_complete and all_complete and not duplicates
    result = {"status": "COMPLETE" if complete else "INCOMPLETE", "run_status": receipt.get("status"),
              "evidence": "UE online closed-loop scripted synthetic Development",
              "controller": "Retained DTR X73 OR observed-depth near-obstacle branch; sources reported separately",
              "truth_definition": "Continuous relative swept declared disc/box proxies with vertical zones; scene floor supplied by capture",
              "body_zone_m": [0.35, 1.75], "foot_trip_proxy_zone_m": [0.006, 0.35],
              "traversable_relief_max_m": 0.006, "injury_truth": False,
              "claim_boundaries": ["No human injury, deployment safety, or generalization claim",
                  "Stopped short of goal is not success", "Latency is measured worker time, not real-time end-to-end guarantee",
                  "No positive risk is not a declaration of safety", "Paired scripts are curated Development controls"],
              "expected_pairs": 8, "loaded_scenarios": len(specs), "catalog_complete": catalog_complete,
              "loaded_episodes": len(episodes), "duplicate_episodes": duplicates,
              "open_loop_contrasts_passed": sum(p["open_loop_contrast_pass"] is True for p in pairs),
              "all_open_loop_contrasts_pass": complete and all(p["open_loop_contrast_pass"] is True for p in pairs),
              "assisted_successes": sum(p["ASSISTED"]["success"] for p in pairs),
              "assisted_complete_denominator": sum(p["ASSISTED"]["status"] == "COMPLETE" for p in pairs),
              "all_assisted_success": complete and all(p["ASSISTED"]["success"] for p in pairs),
              "contact_control_avoided_with_goal_and_causal_motion": sum(
                  p["expected_open_loop_contact"] and p["open_loop_contrast_pass"] is True and
                  p["ASSISTED"]["success"] and p["causal_trajectory"]["trajectory_changed_after_sensor_alert"]
                  for p in pairs), "pairs": pairs}
    reuse_path = root / 'baseline-reuse.json'
    result['execution_reuse'] = (_read(reuse_path) if reuse_path.exists() else
                                 {'reused_open_loop_episodes': 0, 'new_assisted_episodes_expected': len(specs)})
    (root / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, specs, indexed


def render(root, result, specs, indexed):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
        small = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 13)
        title = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 24)
    except OSError:
        font = small = title = ImageFont.load_default()
    payload, gif_frames = [], []
    pairs = {p["scenario_id"]: p for p in result["pairs"]}
    preview = None
    for spec in specs:
        eps = [indexed.get((spec["id"], arm)) for arm in ("OPEN_LOOP", "ASSISTED")]
        available = [ep for ep in eps if ep and ep.get("frames")]
        if not available:
            continue
        maximum = max(ep["frames"][-1]["time_s"] for ep in available)
        times = [min(maximum, i * .4) for i in range(math.ceil(maximum / .4) + 1)]
        images = []
        pair = pairs[spec["id"]]
        for t in times:
            canvas = Image.new("RGB", (1200, 840), "#101b27")
            draw = ImageDraw.Draw(canvas)
            draw.text((22, 15), "WILLOW WALK  |  ONLINE SENSOR -> CONTROL -> MOTION", font=title, fill="#f1f5fa")
            draw.text((22, 50), f'{spec["id"]}   |   shared simulation time {t:.1f}s', font=font, fill="#a5c4e2")
            for side, (arm, ep) in enumerate(zip(("OPEN_LOOP", "ASSISTED"), eps)):
                x0 = 20 + 590 * side
                summary = pair[arm]
                color = "#64dab5" if summary["success"] else "#ffd37e" if summary.get("goal_reached") else "#ff9b91"
                label = arm + (' (reused)' if side == 0 and result['execution_reuse']['reused_open_loop_episodes'] else '')
                draw.text((x0, 84), f'{label} | {summary["status"]} | success={summary["success"]}', font=font, fill=color)
                if not ep or not ep.get("frames"):
                    draw.text((x0 + 12, 220), "INCOMPLETE / NO SENSOR FRAMES", font=font, fill="#ff9b91")
                    continue
                frames = ep["frames"]
                ix = max(0, bisect.bisect_right([f["time_s"] for f in frames], t) - 1)
                frame = frames[ix]
                path = root / "model" / frame.get("rgb_path", "missing")
                if path.is_file():
                    with Image.open(path) as source:
                        rgb = source.convert("RGB").resize((570, 321))
                    canvas.paste(rgb, (x0, 113))
                else:
                    draw.text((x0 + 12, 240), "RGB NOT AVAILABLE", font=font, fill="#ff9b91")
                command = _command(frame)
                prediction = frame.get("response", {}).get("prediction", {})
                sources = "+".join(_sources(frame)) or "NO POSITIVE ALERT"
                terminal = " [terminal frame held]" if t > frames[-1]["time_s"] + .001 else ""
                draw.text((x0, 444), f'sensor t={frame["time_s"]:.1f}s{terminal}', font=small, fill="#c6d5e6")
                draw.text((x0, 467), f'Alert: {sources} | X73 {prediction.get("event", "UNKNOWN")}', font=small, fill="#ffd37e")
                displayed = frame.get('applied_command', {}) if arm == 'OPEN_LOOP' else command
                action_label = 'Applied' if arm == 'OPEN_LOOP' else 'Next'
                draw.text((x0, 490), f'{action_label}: {displayed.get("action", "UNKNOWN")} '
                          f'vx={displayed.get("vx_mps", 0):.2f} vy={displayed.get("vy_mps", 0):.2f} m/s', font=small, fill="#edf3fb")
                draw.text((x0, 513), f'EPISODE END: goal={summary.get("goal_reached", False)} '
                          f'progress={summary.get("forward_progress_m", 0):.2f}m '
                          f'contact={summary.get("contact", "UNKNOWN")}', font=small, fill=color)
                # Evaluator-only top-down panel. Exact declared bounds, not a sensor view.
                top, bottom, left, right = 550, 739, x0 + 10, x0 + 560
                draw.rectangle((left, top, right, bottom), fill="#182a3b", outline="#38536d")
                def xy(x, y):
                    return (left + (x + .5) / 10 * (right - left), (top + bottom) / 2 + y / 5 * (bottom - top))
                p0, p1 = xy(0, 0), xy(8, 0)
                draw.line((*p0, *p1), fill="#476782", width=2)
                for actor in sorted(actors_at(spec, frame["time_s"]), key=lambda a: a["kind"] != "tactile_ground"):
                    if abs(actor["y_m"]) > 2.5:
                        continue
                    px, py = xy(actor["x_m"], actor["y_m"])
                    fill = "#668276" if actor["kind"] == "tactile_ground" else "#ffc46e" if actor["kind"] == "low_obstacle" else "#a6b4c2"
                    if actor["shape"] == "disc":
                        rx, ry = actor["radius_m"] / 10 * (right-left), actor["radius_m"] / 5 * (bottom-top)
                        draw.ellipse((px-rx, py-ry, px+rx, py+ry), fill=fill)
                    else:
                        hx, hy = actor["half_extents_m"]
                        a, b = xy(actor["x_m"]-hx, actor["y_m"]-hy), xy(actor["x_m"]+hx, actor["y_m"]+hy)
                        draw.rectangle((*a, *b), fill=fill)
                trail = [xy(f["ego"]["x_m"], f["ego"]["y_m"]) for f in frames[:ix+1]]
                if len(trail) > 1:
                    draw.line(trail, fill="#52cfef", width=3)
                ex, ey = trail[-1]
                draw.ellipse((ex-8, ey-8, ex+8, ey+8), fill="#52cfef", outline="white", width=2)
                contacts = frame.get("contacts_since_previous", [])
                draw.text((x0, 747), "CONTACT: " + (", ".join(c["contact_type"] for c in contacts) or "none this interval"),
                          font=small, fill="#ff9b91" if contacts else "#a5bdd3")
            draw.text((22, 780), "EVALUATOR MAP: cyan wearer | gray body proxies | amber foot/trip proxy | green 4mm tactile", font=small, fill="#d0dfed")
            draw.text((22, 804), "Scripted synthetic Development. Body/foot proxies are not human injury truth. Stopped short != success.", font=small, fill="#a8bdd1")
            buffer = io.BytesIO()
            canvas.save(buffer, format="JPEG", quality=76)
            images.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())
            gif_frames.append(canvas.resize((900, 630)).quantize(colors=96))
            if preview is None or (spec["id"] == "low_obstacle_collision" and abs(t-4) < .01):
                preview = canvas.copy()
        payload.append({"scenario_id": spec["id"], "times": times, "images": images})
    if preview:
        preview.save(root / "closed-loop-preview.png")
    if gif_frames:
        gif_frames[0].save(root / "replay.gif", save_all=True, append_images=gif_frames[1:], duration=400, loop=0)
    html = '''<!doctype html><meta charset="utf-8"><title>Willow Walk | paired closed-loop</title>
<style>body{margin:24px auto;max-width:1220px;padding:0 16px;background:#101b27;color:#eef5ff;font:16px system-ui}h1{font-size:25px}img{width:100%;border-radius:8px}button,select{padding:10px;background:#233b50;color:#eef5ff;border:1px solid #607f99;border-radius:6px}input{width:65%;vertical-align:middle}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{color:#b8d0e5}</style>
<h1>Willow Walk · 在线闭环配对回放</h1><p>UE 实际 RGB-D → DTR X73 / 观测深度分支 → 控制指令 → 实际轨迹。左：OPEN_LOOP；右：ASSISTED。两侧按仿真时间对齐。</p>
<p id="reuse"></p>
<p><select id="cases"></select> <button id="play">播放 / 暂停</button> <span id="time"></span></p><img id="frame"><p><input id="seek" type="range" min="0" value="0"></p>
<p>受控合成 Development。身体碰撞与脚部绊倒代理分开记录，不是人体伤害真值；未到达目标不能算成功。短片的结束帧会保留显示。</p><details><summary>实测结果、延迟与证据边界</summary><pre id="report"></pre></details>
<script>const data=PAYLOAD;const report=REPORT;let c=0,i=0,playing=false;const frame=document.getElementById('frame'),seek=document.getElementById('seek'),cases=document.getElementById('cases');
document.getElementById('reuse').textContent=report.execution_reuse.reused_open_loop_episodes?'本轮复用 8 个已完成的直行对照，重新执行 8 个辅助分支；不是 16 个全新分支。':'本轮直行与辅助分支均实际执行。';
data.forEach((d,k)=>{const o=document.createElement('option');o.value=k;o.textContent=d.scenario_id;cases.appendChild(o)});
function show(){if(!data.length)return;frame.src=data[c].images[i];seek.max=data[c].images.length-1;seek.value=i;document.getElementById('time').textContent=data[c].times[i].toFixed(1)+' s'}
cases.onchange=()=>{c=+cases.value;i=0;show()};seek.oninput=()=>{i=+seek.value;show()};document.getElementById('play').onclick=()=>playing=!playing;
setInterval(()=>{if(playing&&data.length){i=(i+1)%data[c].images.length;show()}},400);document.getElementById('report').textContent=JSON.stringify(report,null,2);show();</script>'''
    (root / "replay.html").write_text(html.replace("PAYLOAD", json.dumps(payload)).replace("REPORT", json.dumps(result)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    root = args.run.resolve()
    result, specs, indexed = evaluate(root)
    if not args.metrics_only:
        render(root, result, specs, indexed)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
