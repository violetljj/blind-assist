"""Aggregate the event and empty-road results from explicitly listed validation receipts."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize(events, windows):
    counts = {}
    silent = [e for e in events if e["silent"] is True]
    for arm in ("original", "direct", "edge"):
        counts[arm] = {
            "first_encounters": sum(bool(e["hits"][arm]) for e in silent),
            "first_objects": len({(e["site"], obj) for e in silent for obj in e["hits"][arm]}),
            "all_object_supported_encounters": sum(bool(e["hits"][arm]) for e in events),
            "false_windows": sum(bool(w["alarms"][arm]) for w in windows),
            "false_samples": sum(len(w["alarms"][arm]) for w in windows),
        }
    added = [e["id"] for e in silent if e["hits"]["edge"] and not e["hits"]["direct"]]
    lost = [e["id"] for e in silent if e["hits"]["direct"] and not e["hits"]["edge"]]
    # The extension includes the whole direct arm. Equality in number entails identical traces.
    for w in windows:
        if not set(w["alarms"]["direct"]).issubset(w["alarms"]["edge"]):
            raise ValueError("The combined arm lost a direct alert")
    if lost:
        raise ValueError("The combined arm lost a direct event")
    matched = bool(windows) and all(counts["direct"][k]==counts["edge"][k] for k in ("false_windows","false_samples"))
    return dict(approaches=len(events), silent=len(silent), incomplete=sum(e["silent"] is None for e in events),
                objects=len({(e["site"],o) for e in events for o in e["objects"]}),
                silent_objects=len({(e["site"],o) for e in silent for o in e["objects"]}),
                windows=len(windows), samples=sum(len(w["frames"]) for w in windows),arms=counts,
                edge_only=added,direct_only=lost,same_false_cost=matched,
                edge_only_with_direct_silent_throughout=[e["id"] for e in silent if e["id"] in added
                                                         and e["pass_alarm_samples"]["direct"] == 0],
                increment_at_same_cost=len(added)-len(lost) if matched and silent else None)


def aggregate(paths):
    all_events,all_controls,inputs = [],[],[]
    sources, samples = set(),set()
    site_events, site_controls = defaultdict(list),defaultdict(list)
    for path in paths:
        blob = path.read_bytes()
        d = json.loads(blob)
        identity=(d["manifest_sha256"],d["labels_sha256"])
        if identity in sources:
            raise ValueError("Repeated source/label lock")
        sources.add(identity)
        inputs.append(dict(source=d["dataset"], manifest_sha256=identity[0],
                           labels_sha256=identity[1],receipt_sha256=hashlib.sha256(blob).hexdigest()))
        frame_index = {(r["seq"],r["i"]):r for r in d["frames"]}
        for event in d["events"]:
            rows = [frame_index[tuple(k)] for k in event["frames"]]
            e = {**event,"pass_alarm_samples":{
                arm:sum(bool(r["alert"][arm]) for r in rows if r["alert"] is not None)
                for arm in ("original","direct","edge")}}
            all_events.append(e)
            site_events[e["site"]].append(e)
        for w in d["windows"]:
            if not w["complete"]:
                raise ValueError("Partially scored window: report per source before pooling")
            keys={(w["site"], seq, i) for seq,i in w["frames"]}
            if keys & samples:
                raise ValueError("Repeated control observations across receipts")
            samples.update(keys)
            all_controls.append(w)
            site_controls[w["site"]].append(w)
    if len({(e["site"],e["id"]) for e in all_events}) != len(all_events):
        raise ValueError("Repeated physical approach")
    sites = sorted(site_events.keys() | site_controls.keys())
    return dict(receipts=inputs,all=summarize(all_events,all_controls),
                sites={s:summarize(site_events[s],site_controls[s]) for s in sites},
                events=all_events,windows=all_controls)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("receipts",type=Path,nargs="+")
    args=p.parse_args()
    result=aggregate(args.receipts)
    tmp=args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=1,allow_nan=False),encoding="utf-8")
    tmp.replace(args.output)
