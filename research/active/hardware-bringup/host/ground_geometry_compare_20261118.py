"""Pair frozen semantic/geometry arms on the original v1.2 evaluable rows.

Match semantic thresholds exclusively on predefined negative outcomes. Never choose a
threshold from a positive row. All counts are descriptive, from consumed development data.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "artifacts.local/work"
RUN = WORK / "ground-geometry-20261118"
PREV = WORK / "ground-alert-20261118"
WALL = WORK / "sidewall-attribution-20261118"
GRID = ("0.80", "0.85", "0.90", "0.93", "0.95", "0.97", "0.98", "0.99", "0.995", "0.999", "1.01")
BANDS = ("LOW", "BODY", "HEAD")
ARMS = ("baseline", "sem80", "geom", *GRID)
FILES = {"adt": "direction-a-adt-20261118/run/rows_adt_apartment.json",
         "sanpo": "direction-a-confirm-20261117/run/rows_sanpo_chest_seg.json"}


def read(path):
    return json.loads(path.read_bytes())


def codes(bands):
    if set(bands) != set(BANDS) or not all(isinstance(bands[b], bool) for b in BANDS):
        raise ValueError(f"Expected three observed boolean alerts, got {bands}")
    return sum(int(bands[b]) << k for k, b in enumerate(BANDS))


@dataclass
class Cohort:
    source: str
    rows: list
    seq: np.ndarray
    truth: np.ndarray
    dist: np.ndarray
    pred: dict
    events: list
    empty: np.ndarray


def original_reference(source, original):
    if source == "sanpo":
        targets = {x["row"] for x in read(WALL / "sanpo/attribution.json") if x["match"]
                   and x["removed"]["walkable_semantic"] and x["removed"]["reference_below_low"]}
        previous = {r["row"]: r["candidate"] for r in read(PREV / "sanpo/predictions.json")}
        return {k: k for k in targets}, previous
    historical = {(r["seq"], r["t"]): r for seq in sorted({r["seq"] for r in original})
                  for r in read(PREV / "adt" / f"{seq}.json")}
    floor = {(r["seq"], r["i"]) for r in read(WALL / "adt/details.json")
             if r["audit"].get("tof_dav2s", {}).get("eliminated", {}).get("floor_plane_3cm")}
    targets, previous = {}, {}
    event = -1
    last = None
    for k, r in enumerate(original):
        old = historical[r["seq"], r["t"]]
        previous[k] = old["after"]
        if (old["seq"], old["i"]) in floor:
            if (last is None or old["seq"] != last["seq"] or old["i"] != last["i"] + 1
                    or old["t"] - last["t"] > .75):
                event += 1
            targets[k] = event
            last = old
    assert len(targets) == 7 and event == 5
    return targets, previous


def load(source):
    original = read(WORK / FILES[source])
    inputs = read(RUN / source / "rows.json")
    by_row = {int(r["row_index"]): r for r in inputs}
    assert len(by_row) == len(inputs) and set(by_row).issubset(range(len(original)))
    method = "tof_dav2s" if source == "adt" else "hybrid_out"
    targets, previous = original_reference(source, original)
    all_arms = {a: [] for a in ARMS}
    rows, event_rows = [], {}
    for index, r in enumerate(original):
        bands = r["alert"][method]
        item = by_row.get(index)
        if item is not None:
            assert item["seq"] == r["seq"] and abs(item["t"] - r["t"]) < 1.e-6
            assert item["baseline"] == bands
        if not r["evaluable"] or any(v is None for v in bands.values()):
            assert index not in targets
            continue
        baseline = codes(bands)
        if item is None:
            if baseline:
                raise ValueError(f"Missing baseline-positive {source} row {index}")
            values = {a: 0 for a in ARMS}
        else:
            controls = {float(k): v for k, v in item["thresholds"].items()}
            assert set(controls) == set(map(float, GRID))
            values = {a: codes(item[a]) for a in ("baseline", "sem80", "geom")}
            values.update({a: codes(controls[float(a)]) for a in GRID})
        assert values["sem80"] == codes(previous.get(index, bands)) == values["0.80"]
        assert values["1.01"] == baseline
        assert all((values[a] & ~baseline) == 0 for a in ARMS)
        assert all((values[a] & ~values[b]) == 0 for a, b in zip(GRID[:-1], GRID[1:]))
        assert (values["sem80"] & ~values["geom"]) == 0
        if index in targets:
            assert list(r["truth"].values()).count(0) == 3 and baseline
            event_rows.setdefault(targets[index], []).append(len(rows))
        dist = r["truth_dist"]["LOW"]
        rows.append(dict(src=index, seq=r["seq"], t=r["t"], truth=r["truth"], distance=dist))
        for a in ARMS:
            all_arms[a].append(values[a])
    truth = np.array([[r["truth"][b] for b in BANDS] for r in rows], dtype=np.int8)
    c = Cohort(source, rows, np.array([r["seq"] for r in rows]), truth,
               np.array([r["distance"] if r["distance"] is not None else np.inf for r in rows]),
               {a: np.array(v, dtype=np.uint8) for a, v in all_arms.items()},
               list(event_rows.values()), (truth == 0).all(axis=1))
    events, evaluable, empty, before, sem = (6, 1992, 23, 11, 5) if source == "adt" else (94, 1951, 1771, 123, 57)
    assert (len(c.events), len(rows), int(c.empty.sum()), int(np.sum(c.pred["baseline"][c.empty] != 0))) == (
        events, evaluable, empty, before)
    assert negative(c, c.pred["sem80"], np.ones(len(rows), bool))[0] == sem
    return c


def negative(c, pred, mask):
    """Match only on ground false alarms and all strict-empty baseline false alarms."""
    erased_units = sum(1 for event in c.events if mask[event[0]] and not np.any(pred[event]))
    fp = c.empty & mask & (c.pred["baseline"] != 0)
    return erased_units, int(np.sum(fp & (pred == 0)))


def match(c, training):
    target = negative(c, c.pred["geom"], training)
    table = {t: negative(c, c.pred[t], training) for t in GRID}

    def order(t):
        g, e = table[t]
        return abs(g - target[0]), abs(e - target[1]), -float(t)

    # Also show neighbours giving strictly less/more measured ground-FP reduction. An endpoint can be open.
    best = min(GRID, key=order)
    neighbours = {}
    for label, keys in (("less", [t for t in GRID if table[t][0] < target[0]]),
                        ("equal", [t for t in GRID if table[t][0] == target[0]]),
                        ("more", [t for t in GRID if table[t][0] > target[0]])):
        neighbours[label] = min(keys, key=order) if keys else None
    index = GRID.index(best)
    adjacent_grid = list(GRID[max(0, index-1):index]) + list(GRID[index+1:index+2])
    same_ground_levels = {key: [t for t in GRID if table[t][0] == table[representative][0]]
                          if representative else [] for key, representative in neighbours.items()}
    return best, dict(geometry=list(target), controls={t: list(n) for t, n in table.items()},
                      chosen=best, signed_chosen_minus_geometry=[x-y for x, y in zip(table[best], target)],
                      bracket=neighbours, all_thresholds_at_bracket_ground_counts=same_ground_levels,
                      thresholds_adjacent_to_chosen=adjacent_grid,
                      training_ground_units=sum(bool(training[ev[0]]) for ev in c.events),
                      training_strict_empty=int(np.sum(training & c.empty)))


def groups(c):
    lo, body, head = c.truth.T
    strict = (lo == 1) & (body == 0) & (head == 0)
    ground = np.zeros(len(c.rows), bool)
    for ev in c.events:
        ground[ev] = True
    return dict(ground_frames=ground, strict_empty=c.empty, any_positive=(c.truth == 1).any(axis=1),
                any_unknown=~((c.truth == 1).any(axis=1) | c.empty), low_positive=lo == 1,
                low_only_v12=(lo == 1) & (body == 0), strict_low=strict, near_strict_low=strict & (c.dist <= 1.5))


def paired(left, right, mask, bit=0):
    a, b = left[mask], right[mask]
    a = (a != 0) if bit == 0 else ((a & bit) != 0)
    b = (b != 0) if bit == 0 else ((b & bit) != 0)
    return dict(n=len(a), left=int(a.sum()), right=int(b.sum()), only_left=int(np.sum(a & ~b)),
                only_right=int(np.sum(~a & b)))


def compare(c, a, b, mask):
    output = {}
    for label, subset in groups(c).items():
        detectors = (("ANY", 0), ("LOW", 1)) if "low" in label else (("ANY", 0),)
        output[label] = {name: paired(a, b, mask & subset, bit) for name, bit in detectors}
    ea = np.array([any(a[ev]) for ev in c.events if mask[ev[0]]], dtype=bool)
    eb = np.array([any(b[ev]) for ev in c.events if mask[ev[0]]], dtype=bool)
    output["ground_units"] = paired(ea, eb, np.ones(len(ea), bool))
    return output


def main(source):
    c = load(source)
    all_rows = np.ones(len(c.rows), bool)
    chosen, info = match(c, all_rows)
    folds = {}
    held_out = np.empty_like(c.pred["geom"])
    for s in np.unique(c.seq):
        test = c.seq == s
        t, diagnostics = match(c, ~test)
        if diagnostics["training_ground_units"] == 0:
            raise ValueError(f"No ground-FP unit remains for held-out {s}")
        held_out[test] = c.pred[t][test]
        folds[s] = dict(train=diagnostics, test_negative_geom=list(negative(c, c.pred["geom"], test)),
                        test_negative_control=list(negative(c, c.pred[t], test)))
    c.pred.update(matched=c.pred[chosen], loo=held_out)
    vs_baseline = {a: compare(c, c.pred["baseline"], b, all_rows) for a, b in c.pred.items()}
    vs_geometry = {a: compare(c, c.pred["geom"], c.pred[a], all_rows) for a in ("sem80", "matched", "loo", *GRID)}
    cases = []
    for i in np.flatnonzero(groups(c)["strict_low"]):
        cases.append({**c.rows[i], "loo_threshold": folds[c.seq[i]]["train"]["chosen"],
                      "arms": {a: [b for j, b in enumerate(BANDS) if p[i] & (1 << j)] for a, p in c.pred.items()}})
    per_seq = {s: {a: compare(c, c.pred["geom"], c.pred[a], c.seq == s) for a in ("sem80", "matched", "loo")}
               for s in folds}
    for v in ("matched", "loo"):
        for g in ("strict_low", "near_strict_low", "low_positive", "strict_empty", "any_positive"):
            for detector, total in vs_geometry[v][g].items():
                assert total == {k: sum(q[v][g][detector][k] for q in per_seq.values()) for k in total}
    summary = dict(source=source, n=len(c.rows), sequences=len(folds), bit_order=BANDS,
                   matching_key="Absolute ground-FP removals difference, then all strict-empty FP removals "
                       "difference, then highest threshold; negatives only. Count complete original ADT floor "
                       "episodes and SANPO sampled frames. Array order: ground units, all strict-empty FP frames.",
                   selected_using_all_negatives=info, loo_threshold_histogram=dict(Counter(x["train"]["chosen"]
                   for x in folds.values())), baseline_vs_arm=vs_baseline, geometry_vs_arm=vs_geometry)
    if source == "sanpo":
        stone, = [r for r in cases if r["src"] == 1111]
        assert abs(stone["distance"] - 1.70756) < .001
        others = c.seq != stone["seq"]
        summary["outside_known_stone_recording"] = {
            a: compare(c, c.pred["geom"], c.pred[a], others) for a in ("sem80", "matched", "loo")}
        summary["prior_strict_losses_case_ids"] = {
            "unresolved_scaffold_shadow": 0, "raised_stone": 1111, "snow_tree_pit": 1467}
    out = RUN / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    for suffix, obj in (("summary", summary), ("strict_low_cases", cases), ("per_sequence", per_seq),
                        ("loo", folds)):
        (out / f"{source}_{suffix}.json").write_text(json.dumps(obj, indent=1, allow_nan=False), encoding="utf-8")
    print(source, "matched", info, flush=True)
    for g in ("ground_units", "strict_empty", "strict_low", "near_strict_low", "low_positive", "any_positive"):
        print(g, vs_geometry["matched"][g], "held-out", vs_geometry["loo"][g])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", choices=tuple(FILES))
    for name in parser.parse_args().sources:
        main(name)
